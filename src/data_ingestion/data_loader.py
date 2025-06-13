import pandas as pd
import numpy as np
import os
from datetime import datetime


def _ensure_dataframe(df, name="dataframe"):
    """Helper to ensure input is a DataFrame and handle common issues."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame for {name}, got {type(df)}")
    if df.empty:
        print(f"Warning: {name} is empty.")
    return df


def load_market_data(filepath: str) -> pd.DataFrame:
    """
    Loads market data (OHLCV, adjusted prices).
    Assumes data is stored in Parquet for efficiency or CSV.
    Sets 'Date' as index and ensures correct dtypes.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Market data file not found: {filepath}")

    print(f"Loading market data from: {filepath}")
    if filepath.endswith(".parquet"):
        df = pd.read_parquet(filepath)
    elif filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported market data file format: {filepath}")

    # Basic cleaning and type conversion
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    # Ensure numeric columns are numeric (e.g., sometimes prices come as objects)
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return _ensure_dataframe(df, "market data")


def load_ml_features_data(filepath: str) -> pd.DataFrame:
    """
    Loads pre-calculated ML features data.
    This data is expected to be already cleaned and engineered from raw sources.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ML features data file not found: {filepath}")

    print(f"Loading ML features from: {filepath}")
    if filepath.endswith(".parquet"):
        df = pd.read_parquet(filepath)
    elif filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported features data file format: {filepath}")

    df["Date"] = pd.to_datetime(df["Date"])
    # Assuming a multi-index of (Date, Ticker) or similar for features
    if "Ticker" in df.columns:
        df = df.set_index(["Date", "Ticker"]).sort_index()
    else:
        df = df.set_index("Date").sort_index()

    # Ensure all feature columns are numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return _ensure_dataframe(df, "ML features data")


def load_model_predictions(filepath: str) -> pd.DataFrame:
    """
    Loads pre-generated ML model predictions (e.g., target returns, probabilities).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model predictions file not found: {filepath}")

    print(f"Loading model predictions from: {filepath}")
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    if "Ticker" in df.columns:
        df = df.set_index(["Date", "Ticker"]).sort_index()
    else:
        df = df.set_index("Date").sort_index()
    # Ensure prediction column is numeric
    if "prediction" in df.columns:  # Assuming a 'prediction' column
        df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    return _ensure_dataframe(df, "model predictions")


def load_index_constituents(filepath: str) -> pd.DataFrame:
    """
    Loads historical index constituent data (e.g., MSCI, S&P, FTSE).
    This data is crucial for identifying rebalance events.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Index constituents file not found: {filepath}")

    print(f"Loading index constituents from: {filepath}")
    df = pd.read_csv(filepath, parse_dates=["Date"])
    # Ensure date and ticker are correct
    df["Date"] = pd.to_datetime(df["Date"])
    # Assuming columns like 'Index', 'Ticker', 'Weight', 'Effective_Date', 'Announcement_Date'
    # Set index or just sort for ease of use
    df = df.sort_values(["Date", "Index", "Ticker"])
    return _ensure_dataframe(df, "index constituents")


def load_corporate_actions(filepath: str) -> pd.DataFrame:
    """
    Loads corporate action data (e.g., splits, dividends, mergers).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Corporate actions file not found: {filepath}")

    print(f"Loading corporate actions from: {filepath}")
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "Ticker"])
    # Ensure relevant columns like 'Action_Type', 'Factor' (for splits) are present and correct
    return _ensure_dataframe(df, "corporate actions")


def load_etf_flows(filepath: str) -> pd.DataFrame:
    """
    Loads ETF flow data, which can indicate passive fund behavior.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ETF flows file not found: {filepath}")

    print(f"Loading ETF flows from: {filepath}")
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "ETF"])
    # Ensure flow values are numeric
    if "Flow_USD" in df.columns:
        df["Flow_USD"] = pd.to_numeric(df["Flow_USD"], errors="coerce")
    return _ensure_dataframe(df, "ETF flows")


def load_csv_file(filepath, **kwargs):
    """
    Load a CSV file into a pandas DataFrame.
    Ensures that common missing value representations are handled.
    :param filepath: str, path to the CSV file
    :param kwargs: additional arguments for pd.read_csv
    :return: pd.DataFrame
    """
    df = pd.read_csv(
        filepath,
        na_values=["", "NA", "N/A", "null", "None", "-"],
        keep_default_na=True,
        **kwargs,
    )
    return df


# --- Example Usage (for testing within the module or a notebook) ---
if __name__ == "__main__":
    print("--- Testing data_loader.py functions (using mock data paths) ---")

    # Assume these files exist in 'data/processed/' or 'data/raw/'
    # In a real scenario, you'd generate these mock files first or point to real ones.
    MOCK_MARKET_DATA = "data/processed/market_data_cleaned.parquet"
    MOCK_FEATURES_DATA = "data/processed/ml_features_for_backtest.parquet"
    MOCK_PREDICTIONS_DATA = "results/ml_metrics/model_predictions_backtest_period.csv"
    MOCK_INDEX_CONSTITUENTS = "data/processed/index_constituents.csv"
    MOCK_CORP_ACTIONS = "data/processed/corporate_actions.csv"
    MOCK_ETF_FLOWS = "data/processed/etf_flows.csv"

    # --- Create dummy data for testing ---
    # This part would typically be in a separate data generation script or notebook
    # For demonstration, we create them on the fly if they don't exist.
    if not os.path.exists("data/processed"):
        os.makedirs("data/processed")
    if not os.path.exists("results/ml_metrics"):
        os.makedirs("results/ml_metrics")

    # Dummy Market Data
    if not os.path.exists(MOCK_MARKET_DATA):
        dates = pd.date_range(start="2020-01-01", end="2024-12-31", freq="D")
        tickers = ["AAPL", "MSFT", "GOOGL"]
        market_data_dict = {}
        for ticker in tickers:
            prices = np.cumprod(1 + np.random.normal(0.0005, 0.01, len(dates))) * 100
            volumes = np.random.randint(1_000_000, 10_000_000, len(dates))
            df_ticker = pd.DataFrame(
                {
                    "Date": dates,
                    "Ticker": ticker,
                    "Open": prices * 0.99,
                    "High": prices * 1.01,
                    "Low": prices * 0.98,
                    "Close": prices,
                    "Adj Close": prices,
                    "Volume": volumes,
                }
            )
            market_data_dict[ticker] = df_ticker
        dummy_market_data = pd.concat(list(market_data_dict.values())).reset_index(
            drop=True
        )
        dummy_market_data.to_parquet(MOCK_MARKET_DATA, index=False)
        print(f"Created dummy market data at {MOCK_MARKET_DATA}")

    # Dummy Features Data
    if not os.path.exists(MOCK_FEATURES_DATA):
        dates = pd.date_range(start="2020-01-01", end="2024-12-31", freq="D")
        tickers = ["AAPL", "MSFT", "GOOGL"]
        feature_data = []
        for date in dates:
            for ticker in tickers:
                feature_data.append(
                    {
                        "Date": date,
                        "Ticker": ticker,
                        "momentum_1m": np.random.normal(0, 0.05),
                        "volatility_3m": np.random.normal(0.1, 0.02),
                        "news_sentiment_avg": np.random.uniform(-1, 1),
                        "tfqf_implied_vol_diff": np.random.normal(0, 0.01),
                    }
                )
        dummy_features_data = pd.DataFrame(feature_data)
        dummy_features_data.to_parquet(MOCK_FEATURES_DATA, index=False)
        print(f"Created dummy features data at {MOCK_FEATURES_DATA}")

    # Dummy Predictions Data
    if not os.path.exists(MOCK_PREDICTIONS_DATA):
        dates = pd.date_range(start="2020-01-01", end="2024-12-31", freq="D")
        tickers = ["AAPL", "MSFT", "GOOGL"]
        predictions_data = []
        for date in dates:
            for ticker in tickers:
                predictions_data.append(
                    {
                        "Date": date,
                        "Ticker": ticker,
                        "prediction": np.random.normal(
                            0.001, 0.005
                        ),  # Example: predicted daily return
                    }
                )
        dummy_predictions_data = pd.DataFrame(predictions_data)
        dummy_predictions_data.to_csv(MOCK_PREDICTIONS_DATA, index=False)
        print(f"Created dummy predictions data at {MOCK_PREDICTIONS_DATA}")

    # --- Test loading functions ---
    try:
        market_df = load_market_data(MOCK_MARKET_DATA)
        print(f"Loaded market data. Shape: {market_df.shape}")
        print(market_df.head())

        features_df = load_ml_features_data(MOCK_FEATURES_DATA)
        print(f"\nLoaded ML features data. Shape: {features_df.shape}")
        print(features_df.head())

        predictions_df = load_model_predictions(MOCK_PREDICTIONS_DATA)
        print(f"\nLoaded model predictions. Shape: {predictions_df.shape}")
        print(predictions_df.head())

        # For other data types, you'd create similar dummy files and test
        # print("\n(To test other data types, ensure their dummy files exist)")
        # index_constituents_df = load_index_constituents(MOCK_INDEX_CONSTITUENTS)
        # corp_actions_df = load_corporate_actions(MOCK_CORP_ACTIONS)
        # etf_flows_df = load_etf_flows(MOCK_ETF_FLOWS)

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure mock data files exist or create them.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print("\n--- data_loader.py testing complete ---")
