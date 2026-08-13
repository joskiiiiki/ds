import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import polars as p
    import numpy as np
    import optuna as opt
    import seaborn as sb
    import matplotlib.pyplot as plt
    import matplotlib
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import TargetEncoder, FunctionTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import HistGradientBoostingRegressor


@app.cell
def _():
    df = p.read_csv("train.csv", infer_schema_length=None)

    cat_cols = [
        "MSSubClass",
        "Street",
        "MSZoning",
        "Alley",
        "LotShape",
        "LandContour",
        "Utilities",
        "LotConfig",
        "LandSlope",
        "Neighborhood",
        "Condition1",
        "Condition2",
        "BldgType",
        "HouseStyle",
        "RoofStyle",
        "RoofMatl",
        "Exterior1st",
        "Exterior2nd",
        # "ExterQual", "ExterCond"
        "MasVnrType",
        "Foundation",
        # "BsmtQual",
        # "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "Heating",
        # "HeatingQC",
        "CentralAir",
        "Electrical",
        # "KitchenQual",
        "Functional",
        # "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "PavedDrive",
        "PoolQC",
        "Fence",
        "MiscFeature",
        "SaleType",
        "SaleCondition"
    ]
    base_num_cols = [
      "LotFrontage",
      "LotArea",
      "OverallQual",
      "OverallCond",
      "YearBuilt",
      "YearRemodAdd",
      "MasVnrArea",
      "BsmtFinSF1",
      "BsmtFinSF2",
      "BsmtUnfSF",
      "TotalBsmtSF",
      "1stFlrSF",
      "2ndFlrSF",
      "LowQualFinSF",
      "GrLivArea",
      "BsmtFullBath",
      "BsmtHalfBath",
      "FullBath",
      "GarageYrBlt",
      "HalfBath",
      "BedroomAbvGr",
      "KitchenAbvGr",
      "TotRmsAbvGrd",
      "Fireplaces",
      "GarageCars",
      "GarageArea",
      "WoodDeckSF",
      "OpenPorchSF",
      "EnclosedPorch",
      "3SsnPorch",
      "ScreenPorch",
      "PoolArea",
      "MiscVal",
      "MoSold",
      "YrSold",
    ]

    num_cols = base_num_cols + [
        "TotalArea",
        "ExterQual", "ExterCond", "BsmtQual", "HeatingQC", "KitchenQual", "FireplaceQu"
    ]

    qual_map: dict[str, int] = {
        "Ex": 4,
        "Gd": 3,
        "TA": 2,
        "Fa": 1,
        "Po": 0,
    }

    def prepare_df(d: p.DataFrame, is_train: bool=False) -> p.DataFrame:
        d = d.with_columns(
                p.col(base_num_cols).cast(p.Float32, strict=False)
            )
        d = d.with_columns(
            d.select(
                (p.col("TotalBsmtSF") + p.col("GrLivArea")).alias("TotalArea"),
                p.col("ExterQual", "ExterCond", "BsmtQual", "HeatingQC", "KitchenQual", "FireplaceQu").replace(qual_map).cast(p.Float32, strict=False),
                (2010 - p.col("YearBuilt")).alias("YearBuilt"),
                (2010 - p.col("YearRemodAdd")).alias("YearRemodAdd"),
            ),
        )
        if is_train:
            d = d.filter(
            ~ ((p.col("GrLivArea") > 4000) & (p.col("SalePrice") < 200000))
            )
            
        return d.select(p.col(cat_cols + num_cols + (["SalePrice"] if "SalePrice" in d.columns else [])))

    df_clean = prepare_df(df, is_train=True)

    mo.ui.table(df_clean, max_columns=None)
    return cat_cols, df, df_clean, num_cols, prepare_df


@app.cell
def _(df):
    plt.hist(df["SalePrice"].log1p())
    return


@app.cell
def _():
    return


@app.cell
def _(df):
    corr = (
        df.select(p.col(p.Float64, p.Int64))
        .to_pandas()
        .corr()["SalePrice"]
        .sort_values()
    )
    colors = ["red" if abs(c) >= 0.5 else "grey" for c in corr]
    plt.xticks(rotation=-90)
    sb.barplot(corr, palette=colors)
    return


@app.cell
def _(cat_cols, df_clean):
    stats = [
        p.col("SalePrice").mean().alias("mean"),
        p.col("SalePrice").std().alias("std"),
    ]

    def vis(ax, col, df):
        ax.set_title(col)
        if col in cat_cols:
            s = df.group_by(col).agg(stats).sort("mean")
            ax.bar(s[col], s["mean"] / 1000)
            ax.errorbar(s[col], s["mean"] / 1000, s["std"] * 1.645 / 1000, fmt="none", color="red")
        else:
            ax.scatter(df[col], df["SalePrice"] / 1000, alpha=0.3, s=5)

    all_cols = [c for c in df_clean.columns if c != "SalePrice"]
    n = len(all_cols)
    cols_per_row = 6
    rows = (n + cols_per_row - 1) // cols_per_row

    fig, axes = plt.subplots(rows, cols_per_row, figsize=(20, rows * 3))
    for ax, col in zip(axes.flatten(), all_cols):
        vis(ax, col, df_clean)
    for ax in axes.flatten()[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig("categories.png", dpi=300)
    fig
    return


@app.cell
def _(cat_cols, df, df_clean):
    cat_cols_idx = [df_clean.get_column_index(c) for c in cat_cols]

    X = df_clean.drop("SalePrice").to_pandas()
    y = df_clean["SalePrice"].to_pandas()

    X_t, X_e, y_t, y_e = train_test_split(X, y, shuffle=True, train_size=0.8)

    df.select(p.col(p.String)).columns
    return X, X_e, X_t, cat_cols_idx, y, y_e, y_t


@app.cell
def _(cat_cols, num_cols):
    preprocessor = ColumnTransformer([
        ("cat", "passthrough", cat_cols), 
        ("num", SimpleImputer(strategy="mean"), num_cols),
    ])  # kategorische Spalten durchreichen
    return (preprocessor,)


@app.cell(disabled=True)
def _(X_t, cat_cols_idx, preprocessor, y_t):

    y_t_log = np.log1p(y_t)

    def objective(trial: opt.Trial, X, y) -> float:
        pipe = Pipeline([
            ("pre", preprocessor),
            ("model", HistGradientBoostingRegressor(
                categorical_features=cat_cols_idx,
                learning_rate=trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
                l2_regularization=trial.suggest_categorical("l2_regularization", [0.0, 0.1, 1.0, 10.0])
            ))
        ])

        pipe.fit(X_t, y_t_log)

        return cross_val_score(pipe, X, y, cv=5).mean()

    study = opt.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_t, y_t_log), 100, n_jobs=4)
    return objective, study


@app.cell
def _(X_e, X_t, cat_cols_idx, preprocessor, study, y_e, y_t):
    pipe = Pipeline([
        ("pre", preprocessor),
        ("model", HistGradientBoostingRegressor(
            categorical_features=cat_cols_idx,
            **study.best_params
        ))
    ])

    pipe.fit(X_t, np.log1p(y_t))
    pipe.score(X_e, np.log1p(y_e))
    return (pipe,)


@app.cell(disabled=True)
def _(X, cat_cols_idx, objective, pipe, prepare_df, preprocessor, study, y):

    study.optimize(lambda trial: objective(trial, X, np.log1p(y)), 100, n_jobs=4)

    pipe_comp = Pipeline([
        ("pre", preprocessor),
        ("model", HistGradientBoostingRegressor(
            categorical_features=cat_cols_idx,
            **study.best_params
        ))
    ])

    pipe_comp.fit(X, np.log1p(y))

    df_comp = p.read_csv("test.csv", infer_schema_length=None)

    df_comp_clean = prepare_df(df_comp)

    X_comp = df_comp_clean.to_pandas()

    preds = p.DataFrame(
        {
            "Id": df_comp["Id"],
            "SalePrice": np.expm1(pipe.predict(X_comp))
        }
    )

    preds.write_csv("sub.csv")

    return (df_comp,)


@app.cell
def _(df_comp):
    df_comp
    return


if __name__ == "__main__":
    app.run()
