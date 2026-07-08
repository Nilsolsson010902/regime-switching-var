import yfinance as yf
import pandas as pd

def load_market_data(ticker = "^GSPC", start="2000-01-01", end=None):
    """
    Download and return daily market data from Yahoo Finance.

    Returns a cleaned DataFrame with a DatetimeIndex and standard column names.
    """

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress = False)

    # Flatten MultiIndex columns returned by yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, axis=1, level="Ticker")

    # Ensure chronological order
    df = df.sort_index()
    df.index.name = "Date"

    # Remove metadata inherited from MultiIndex
    df.columns.name = None
    return df