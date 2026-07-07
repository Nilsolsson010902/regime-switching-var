import yfinance as yf
import pandas as pd

def load_market_data(ticker = "^GSPC", start="2000-01-01", end=None):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress = False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, axis=1, level="Ticker")

    df = df.sort_index()
    df.index.name = "Date"
    df.columns.name = None
    return df