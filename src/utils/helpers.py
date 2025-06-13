import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import sys


def plot_equity_curve(series, title="Equity Curve"):
    """
    Plot the equity curve.
    :param series: pd.Series or np.ndarray, indexed by date or step
    :param title: str, plot title
    """
    plt.figure(figsize=(10, 5))
    plt.plot(series, label="Equity Curve")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Calculate annualized Sharpe ratio.
    :param returns: np.ndarray or pd.Series, periodic returns
    :param risk_free_rate: float, risk-free rate per period
    :param periods_per_year: int, number of periods per year
    :return: float, Sharpe ratio
    """
    excess_returns = returns - risk_free_rate
    mean = np.mean(excess_returns)
    std = np.std(excess_returns)
    if std == 0:
        return np.nan
    sharpe = (mean / std) * np.sqrt(periods_per_year)
    return sharpe


def calculate_max_drawdown(equity_curve):
    """
    Calculate the maximum drawdown of an equity curve.
    :param equity_curve: pd.Series or np.ndarray, portfolio values
    :return: float, max drawdown (as a negative number)
    """
    equity_curve = np.asarray(equity_curve)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return np.min(drawdown)


def setup_logging(log_file=None, level=logging.INFO):
    """
    Set up logging for the project.
    :param log_file: str or None, if provided logs to file, else to stdout
    :param level: logging level
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers
    )


def validate_data_schema(df, schema):
    """
    Validate DataFrame columns and dtypes against a schema.
    :param df: pd.DataFrame
    :param schema: dict, column name -> dtype (e.g., {'date': 'datetime64[ns]', 'price': 'float'})
    :return: bool, True if valid, raises ValueError otherwise
    """
    for column, dtype in schema.items():
        if column not in df.columns:
            raise ValueError(f"Missing column: {column}")
        if not np.issubdtype(df[column].dtype, np.dtype(dtype).type):
            raise ValueError(
                f"Incorrect dtype for column {column}: expected {dtype}, got {df[column].dtype}"
            )
    return True
