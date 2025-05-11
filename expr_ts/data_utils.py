from datetime import datetime

from loguru import logger
import pandas as pd


def prepare_formated_data(start_date, end_date, stock_datapath="data/exteral/stock.parquet.gz"):

    df = pd.read_parquet(f"{stock_datapath}", engine="pyarrow")
    df = df[(df.index.astype("str") >= start_date) & (df.index.astype("str") <= end_date)]
    df = df.T.reset_index().melt(
        id_vars=["Ticker", "Price"],
        value_vars=list(df.T.columns),
        var_name="Date",
        value_name="value",
    )
    df = df.pivot(index=["Ticker", "Date"], columns="Price", values=df.columns[3:])
    df.columns = df.columns.droplevel(0)
    df.reset_index(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df["DateIdx"] = (
        pd.to_datetime(df["Date"]) - datetime.strptime("2023-01-01", "%Y-%m-%d")
    ).apply(lambda x: x.days)
    logger.info(f"Data prepared: Data shape: {df.shape}")
    return df
