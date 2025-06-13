import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def calculate_momentum(df, price_col="close", lookbacks=[5, 10, 21]):
    """
    Calculate momentum as percent change over various lookback periods.
    Returns a DataFrame with new columns for each lookback.
    """
    result = pd.DataFrame(index=df.index)
    for lb in lookbacks:
        col_name = f"momentum_{lb}"
        result[col_name] = df[price_col].pct_change(lb)
    return result


def calculate_volatility(df, price_col="close", windows=[5, 21]):
    """
    Calculate rolling volatility (std of returns) for given windows.
    Returns a DataFrame with new columns for each window.
    """
    returns = df[price_col].pct_change()
    result = pd.DataFrame(index=df.index)
    for w in windows:
        col_name = f"volatility_{w}"
        result[col_name] = returns.rolling(window=w).std()
    return result


def calculate_moving_average_crossovers(
    df, price_col="close", short_window=10, long_window=50
):
    """
    Calculate moving average crossovers: short MA minus long MA.
    Returns a DataFrame with short_ma, long_ma, and crossover columns.
    """
    result = pd.DataFrame(index=df.index)
    result["short_ma"] = df[price_col].rolling(window=short_window).mean()
    result["long_ma"] = df[price_col].rolling(window=long_window).mean()
    result["ma_crossover"] = result["short_ma"] - result["long_ma"]
    return result


def calculate_on_balance_volume(df, price_col="close", volume_col="volume"):
    """
    Calculate On-Balance Volume (OBV).
    Returns a Series.
    """
    obv = [0]
    for i in range(1, len(df)):
        if df[price_col].iloc[i] > df[price_col].iloc[i - 1]:
            obv.append(obv[-1] + df[volume_col].iloc[i])
        elif df[price_col].iloc[i] < df[price_col].iloc[i - 1]:
            obv.append(obv[-1] - df[volume_col].iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index, name="on_balance_volume")


def calculate_volume_trend(df, volume_col="volume", window=20):
    """
    Calculate rolling mean and z-score of volume.
    Returns a DataFrame with avg_volume and volume_zscore columns.
    """
    avg_volume = df[volume_col].rolling(window=window).mean()
    std_volume = df[volume_col].rolling(window=window).std()
    zscore = (df[volume_col] - avg_volume) / (std_volume + 1e-8)
    return pd.DataFrame(
        {
            f"avg_volume_{window}": avg_volume,
            f"volume_zscore_{window}": zscore,
        },
        index=df.index,
    )


def calculate_average_daily_volume(df, volume_col="volume", window=21):
    """
    Calculate average daily volume over a rolling window.
    Returns a Series.
    """
    return df[volume_col].rolling(window=window).mean().rename(f"adv_{window}")


def calculate_bid_ask_spread_proxy(df, high_col="high", low_col="low"):
    """
    Calculate a proxy for bid-ask spread using high/low prices.
    Returns a Series.
    """
    spread = (df[high_col] - df[low_col]) / ((df[high_col] + df[low_col]) / 2)
    return spread.rename("bid_ask_spread_proxy")


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
    return scaled_df
