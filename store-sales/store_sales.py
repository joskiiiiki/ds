import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as p
    import matplotlib.pyplot as plt
    import seaborn as sb

    return p, plt


@app.cell
def _(p):
    df = p.read_csv("train.csv")

    df = df.with_columns(df.select(p.col("date").cast(p.Date))).sort("date")

    sales = df.group_by(p.col("date")).agg(p.col("sales").sum()).sort("date")
    avg_sales = sales["sales"].rolling_mean(window_size=90, center=False, min_samples=90 // 2)
    avg_sales
    return avg_sales, sales


@app.cell
def _(avg_sales, plt, sales):
    def _():
        fig, ax = plt.subplots()
        ax.plot(sales["date"], avg_sales)

        return fig

    _()
    return


if __name__ == "__main__":
    app.run()
