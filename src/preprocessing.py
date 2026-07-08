from src import data_loader as dl
import pandas as pd
import numpy as np


def preprocess_market_data(raw_df): 
    """
    Prepare raw market data for modelling.

    - Keep only closing prices
    - Compute daily log returns
    - Remove rows with missing values
    """
    market_df = raw_df[['Close']].copy()
    market_df['Return']  = np.log(market_df['Close'] / market_df['Close'].shift())
    market_df = market_df.dropna()
    return market_df


if __name__ == "__main__":
    raw_df = dl.load_market_data()
    market_df = preprocess_market_data(raw_df)
    print(market_df.head())