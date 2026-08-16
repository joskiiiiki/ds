import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as p
    import matplotlib.pyplot as plt
    import seaborn as sb
    from statsmodels.tsa.deterministic import DeterministicProcess
    from sklearn.linear_model import LinearRegression

    return DeterministicProcess, LinearRegression, p, plt


@app.cell
def _(p):
    df = p.read_csv("train.csv")

    df = df.with_columns(df.select(p.col("date").cast(p.Date))).sort("date")

    sales = df.group_by(p.col("date")).agg(p.col("sales").sum()).sort("date")
    avg_sales = sales["sales"].rolling_mean(window_size=90, center=False, min_samples=90 // 2)

    return avg_sales, sales


@app.cell
def _(avg_sales, plt, sales):
    def _():
        fig, ax = plt.subplots()
        ax.plot(sales["date"], avg_sales)

        return fig

    _()
    return


@app.cell
def _(DeterministicProcess, sales):
    dp = DeterministicProcess(
        sales["date"].to_pandas(),
        constant=True,
        order=1,
        drop=True
    )

    X = dp.in_sample()
    X
    return X, dp


@app.cell
def _(LinearRegression, X, sales):
    t_model = LinearRegression(fit_intercept=False)

    t_model.fit(X, y=sales["sales"])
    t_model.predict(X)
    return (t_model,)


@app.cell
def _(X, avg_sales, dp, plt, sales, t_model):
    def _():
        fig, ax = plt.subplots()
        y = t_model.predict(X)
        ax.set_ylim(0, y.max())
        ax.plot(X.index, y)
        ax.scatter(X.index, sales["sales"], c="grey", s=0.5)
        ax.plot(X.index, avg_sales)
        X_pred = dp.out_of_sample(100)
        y = t_model.predict(X_pred)
        ax.plot(X_pred.index, y)
        return fig
    _()
    dp.out_of_sample(100)
    return


@app.cell
def _(sales):
    lag_f = sales.shift(1)
    return


if __name__ == "__main__":
    app.run()
