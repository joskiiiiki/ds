import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")

with app.setup:
    import polars as p
    import sklearn as sk
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import cross_val_score
    import optuna



@app.cell
def _():

    df_train = p.read_csv("train.csv")
    df_comp = p.read_csv("test.csv")

    def prep_pipe(df: p.DataFrame) -> p.DataFrame:
        return df.with_columns(
            df.select(
                (p.col("Sex") == "female").cast(p.Int32),
                p.col("Age").fill_null(p.col("Age").median()),
                # (p.col("Name").str.contains_any(("Mr.", "Mrs.")).cast(p.Int32)).alias("Married"),
                (p.col("SibSp") + p.col("Parch")).alias("FamSize"),
            
                p.col("Name")
                .str.extract(r",\s*(\w+)\.")
                .fill_null("Rare")
                .cast(p.Categorical)
                .to_physical()
                .alias("Title"),
                p.col("Embarked")
                .fill_null("Elsewhere")
                .cast(p.Categorical)
                .to_physical(),

                p.col("Ticket")
                .str.extract(r"^(.*?)\s*\d+$")
                .replace("", "Regular")
                .cast(p.Categorical)
                .to_physical()
                .alias("TicketPrefix"),

                ((p.col("Sex") == "female") & (p.col("Pclass") == 1)).cast(p.Int32).alias("FemaleFirstClass")
                # ((p.col("Age").mod(1) != 0.0) | p.col("Age").is_null()).cast(p.Int32).alias("UncertainAge")
                # p.col("Fare").qcut(5).to_physical()
                # (p.col("Parch").cut([0, 3]))
            )
        ).drop(p.col("Name", "Cabin", "SibSp", "Parch", "Ticket"))

    df_clean = prep_pipe(df_train)
    X = df_clean.drop("Survived", "PassengerId").to_numpy()
    y = df_clean["Survived"].to_numpy()

    df_comp_clean = prep_pipe(df_comp)
    X_comp = df_comp_clean.drop("PassengerId").to_numpy()
    Id_comp = df_comp_clean["PassengerId"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    df_clean
    return Id_comp, X, X_comp, X_train, X_val, df_train, y, y_train, y_val


@app.cell(disabled=True, hide_code=True)
def _(df_train):

    df_mod = df_train.with_columns()

    col = "UncertainAge"

    metric = (
        df_mod.group_by(p.col(col), p.col("Survived"))
        .len()
        .with_columns((p.col("len") / p.col("len").sum().over(col)).alias("ratio"))
        .sort(col, "Survived")
    )

    survived = metric.filter(p.col("Survived") == 1).sort(col)

    fig, ax = plt.subplots()

    ax.bar(survived[col], survived["ratio"])
    return


@app.cell
def _(X_train, y_train):



    def rf_objective(trial: optuna.Trial) -> float:
        rf = RandomForestClassifier(random_state=42,
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 50),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 30),
            max_features=trial.suggest_float("max_features", 0.4, 1.0),
        )

        rf.fit(X_train, y_train)
        return cross_val_score(rf, X_train, y_train, cv=5).mean()

    rf_study = optuna.create_study(direction="maximize")
    rf_study.optimize(rf_objective, n_jobs=4, n_trials=100, show_progress_bar=True)

    return


@app.cell
def _(Id_comp, X, X_comp, study, y):

    rf = RandomForestClassifier(random_state=42,
        max_leaf_nodes=study.best_params["max_leaf_nodes"],
        min_samples_leaf=study.best_params["min_samples_leaf"],
        max_features=study.best_params["max_features"],
    )

    rf.fit(X, y)

    rf_preds = rf.predict(X_comp)
    rf_submission = p.DataFrame({
        "PassengerId": Id_comp,
        "Survived": rf_preds
    })
    rf_submission.write_csv("rf_submission.csv", include_header=True)
    rf_submission
    return


@app.cell
def _(X_train, y_train):
    def objective(trial: optuna.Trial) -> float:
        xg = HistGradientBoostingClassifier(
            random_state=42,
        
            max_iter=200,
            categorical_features=[0, 1, 4, 6, 7, 8],
            learning_rate=trial.suggest_float("lr", 0.01, 0.1, log=True),
            max_features=trial.suggest_float("max_features", 0.4, 1.0),
            l2_regularization=trial.suggest_categorical("l2_reg", [0.1, 1.0, 10.0]),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 50),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 30)
        
        )
        xg.fit(X_train, y_train)

        return cross_val_score(xg, X_train, y_train, cv=5).mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, show_progress_bar=True, n_jobs=4)
    return (study,)


@app.cell
def _(Id_comp, X, X_comp, study, y):
    xg = HistGradientBoostingClassifier(
        random_state=42,
    
        max_iter=200,
        categorical_features=[0, 1, 4, 6, 7, 8],
        learning_rate=study.best_params["lr"],
        max_features=study.best_params["max_features"],
        l2_regularization=study.best_params["l2_reg"],
        max_leaf_nodes=study.best_params["max_leaf_nodes"],
        min_samples_leaf=study.best_params["min_samples_leaf"]
    
    )
    xg.fit(X, y)
    preds = xg.predict(X_comp)
    submission = p.DataFrame({
        "PassengerId": Id_comp,
        "Survived": preds
    })
    submission.write_csv("submission.csv", include_header=True)
    submission
    study.best_params

    return


@app.cell(disabled=True)
def _(X_train, X_val, y_train, y_val):
    def log_reg_objective(trial: optuna.Trial) -> float:
        log_reg = LogisticRegression(
            random_state=42, 
            max_iter=trial.suggest_int("max_iter", 100, 1000, log=True), 
            solver=trial.suggest_categorical("solver", ["lbfgs", "liblinear", "newton-cholesky", "newton-cg"]), 
            C=trial.suggest_float("C", 0.001, 1.0, log=True)
        )

        log_reg.fit(X_train, y_train)

        return cross_val_score(log_reg, X_val, y_val, cv=5).mean()

    log_reg_study = optuna.create_study(direction="maximize")
    log_reg_study.optimize(log_reg_objective, n_trials=1000, show_progress_bar=True)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
