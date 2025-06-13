import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def calculate_momentum(df, lookback_period=10, price_col="close"):
    """
    Calculate momentum as the percent change over a lookback period.
    :param df: pd.DataFrame with a price column
    :param lookback_period: int, number of periods to look back
    :param price_col: str, column name for price
    :return: pd.Series, momentum values
    """
    return (
        df[price_col]
        .pct_change(periods=lookback_period)
        .rename(f"momentum_{lookback_period}")
    )


def compute_volatility(df, lookback_period=10, price_col="close"):
    """
    Calculate rolling volatility (standard deviation of returns).
    :param df: pd.DataFrame with a price column
    :param lookback_period: int, window size
    :param price_col: str, column name for price
    :return: pd.Series, volatility values
    """
    returns = df[price_col].pct_change()
    return (
        returns.rolling(window=lookback_period)
        .std()
        .rename(f"volatility_{lookback_period}")
    )


def compute_volume_profile(df, lookback_period=20, volume_col="volume"):
    """
    Calculate rolling average and z-score of volume.
    :param df: pd.DataFrame with a volume column
    :param lookback_period: int, window size
    :param volume_col: str, column name for volume
    :return: pd.DataFrame with 'avg_volume' and 'volume_zscore'
    """
    avg_volume = df[volume_col].rolling(window=lookback_period).mean()
    std_volume = df[volume_col].rolling(window=lookback_period).std()
    zscore = (df[volume_col] - avg_volume) / (std_volume + 1e-8)
    return pd.DataFrame(
        {
            f"avg_volume_{lookback_period}": avg_volume,
            f"volume_zscore_{lookback_period}": zscore,
        }
    )


def derive_news_sentiment_features(sentiment_data, lookback_period=5):
    """
    Aggregate news sentiment scores over a rolling window.
    :param sentiment_data: pd.DataFrame with 'date' and 'sentiment_score'
    :param lookback_period: int, window size
    :return: pd.Series, rolling mean sentiment
    """
    sentiment_data = sentiment_data.sort_values("date")
    return (
        sentiment_data["sentiment_score"]
        .rolling(window=lookback_period)
        .mean()
        .rename(f"sentiment_mean_{lookback_period}")
    )


def apply_scaling(df, method="standard", columns=None):
    """
    Apply feature scaling to selected columns.
    :param df: pd.DataFrame
    :param method: 'standard' or 'minmax'
    :param columns: list of columns to scale, or None for all numeric
    :return: pd.DataFrame with scaled features
    """
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    scaled = scaler.fit_transform(df[columns])
    scaled_df = df.copy()
    scaled_df[columns] = scaled
    return
