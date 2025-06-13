import numpy as np
import pandas as pd
from .backtester import Backtester


class MonteCarloBacktester:
    def __init__(self, backtester_config=None):
        self.backtester_config = backtester_config or {}

    def run_monte_carlo_simulations(
        self, num_paths, path_generator, strategy_signals, market_data, config=None
    ):
        """
        Run multiple backtests on synthetic market paths.
        :param num_paths: int, number of Monte Carlo simulations
        :param path_generator: function, generates synthetic market_data for each simulation
        :param strategy_signals: pd.DataFrame, signals to use for all paths (or a function to generate signals per path)
        :param market_data: pd.DataFrame, historical market data (used as base for path generation)
        :param config: dict, optional config for each backtest
        :return: dict with keys 'equity_curves' (list of pd.DataFrame), 'metrics' (list of dicts)
        """
        equity_curves = []
        metrics_list = []

        for i in range(num_paths):
            synthetic_market_data = path_generator(market_data)
            # If strategy_signals is a function, generate signals per path
            if callable(strategy_signals):
                signals = strategy_signals(synthetic_market_data)
            else:
                signals = strategy_signals

            backtester = Backtester(**self.backtester_config)
            equity_curve, metrics = backtester.run_backtest(
                signals, synthetic_market_data, config
            )
            equity_curves.append(equity_curve)
            metrics_list.append(metrics)

        return {"equity_curves": equity_curves, "metrics": metrics_list}

    @staticmethod
    def simple_gbm_path_generator(market_data, mu=None, sigma=None, seed=None):
        """
        Generate synthetic market paths using Geometric Brownian Motion (GBM).
        :param market_data: pd.DataFrame, historical market data (MultiIndex columns: symbol, field)
        :param mu: dict or float, drift per symbol or global
        :param sigma: dict or float, volatility per symbol or global
        :param seed: int, random seed
        :return: pd.DataFrame, synthetic market data with same shape as input
        """
        np.random.seed(seed)
        symbols = market_data.columns.get_level_values(0).unique()
        fields = market_data.columns.get_level_values(1).unique()
        dates = market_data.index

        synthetic_data = pd.DataFrame(index=dates, columns=market_data.columns)
        for symbol in symbols:
            close_prices = market_data[(symbol, "close")].values
            if mu is None:
                mu_ = np.mean(np.diff(np.log(close_prices)))
            else:
                mu_ = mu[symbol] if isinstance(mu, dict) else mu
            if sigma is None:
                sigma_ = np.std(np.diff(np.log(close_prices)))
            else:
                sigma_ = sigma[symbol] if isinstance(sigma, dict) else sigma

            S0 = close_prices[0]
            n = len(close_prices)
            dt = 1.0
            rand = np.random.normal(0, 1, n - 1)
            log_returns = (mu_ - 0.5 * sigma_**2) * dt + sigma_ * np.sqrt(dt) * rand
            log_prices = np.concatenate(
                [[np.log(S0)], np.log(S0) + np.cumsum(log_returns)]
            )
            synthetic_close = np.exp(log_prices)

            synthetic_data[(symbol, "close")] = synthetic_close

            # Copy other fields as-is or fill with NaN
            for field in fields:
                if field != "close":
                    synthetic_data[(symbol, field)] = np.nan

        return synthetic_data
