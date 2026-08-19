import numpy as np
import pandas as pd
from .garch import GarchOutput


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    Compute the daily log returns of a price series.
    """
    validate_feature_input(prices)

    if (prices <= 0).any():
        raise ValueError("Prices must be strictly positive to compute log returns.")

    log_returns = np.log(prices / prices.shift(1))
    log_returns.name = "Returns"

    return log_returns


def compute_momentum(prices: pd.Series, window: int = 20) -> pd.Series:
    """
    Compute log momentum over a specified window.

    Parameters
    ----------
    prices:
        A pandas Series of prices.
    window:
        The number of periods used in the momentum calculation.

    Returns
    -------
    pd.Series
        Log price change over the specified window.
    """
    validate_feature_input(prices)

    if window <= 0:
        raise ValueError("Window must be a positive integer.")

    momentum = np.log(prices / prices.shift(window))
    momentum.name = f"Momentum_{window}"

    return momentum


def compute_rolling_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Compute rolling standard deviation of a return series.

    Parameters
    ----------
    returns:
        A pandas Series of returns.
    window:
        The number of periods used in the rolling calculation.

    Returns
    -------
    pd.Series
        Rolling volatility over the specified window.
    """
    validate_feature_input(returns)

    if window <= 0:
        raise ValueError("Window must be a positive integer.")

    rolling_volatility = returns.rolling(window=window).std()
    rolling_volatility.name = f"Rolling_Volatility_{window}"

    return rolling_volatility


def validate_feature_input(feature_series: pd.Series) -> None:
    """
    Validate a pandas Series used for feature construction.

    Missing values are allowed because lagged and rolling transformations
    naturally produce NaN values at the beginning of the series.
    """
    if not isinstance(feature_series, pd.Series):
        raise TypeError("Input must be a pandas Series.")

    if feature_series.empty:
        raise ValueError("Input Series is empty.")

    if not pd.api.types.is_numeric_dtype(feature_series):
        raise TypeError("Input Series must contain numeric values.")

    if feature_series.index.has_duplicates:
        raise ValueError("Input Series index contains duplicate values.")


def build_hmm_features(ticker: pd.DataFrame, garch: GarchOutput, window: int = 20) -> pd.DataFrame:
    """
    Build a modelling-ready feature DataFrame for HMM analysis.

    The initial feature set contains:
    - daily log returns,
    - log momentum,
    - GARCH conditional volatility.
    """
    if not isinstance(ticker, pd.DataFrame):
        raise TypeError("Ticker data must be a pandas DataFrame.")

    if "Close" not in ticker.columns:
        raise ValueError("Ticker DataFrame must contain a 'Close' column.")

    prices = ticker["Close"].astype(float)

    log_returns = compute_log_returns(prices)
    momentum = compute_momentum(prices, window=window)

    # GARCH was fitted using returns in percentage units.
    # Divide by 100 to align volatility with decimal-form returns.
    conditional_volatility = (garch.conditional_volatility / 100).rename("Conditional_Volatility")

    hmm_features = pd.concat(
        [
            log_returns,
            momentum,
            conditional_volatility
        ],
        axis=1
    )

    hmm_features = (hmm_features.replace([np.inf, -np.inf], np.nan).dropna().sort_index())
    return hmm_features