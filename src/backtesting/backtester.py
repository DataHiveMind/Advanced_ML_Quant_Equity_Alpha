import numpy as np
import pandas as pd


class Backtester:
    def __init__(self, initial_cash=1_000_000, transaction_cost=0.001, slippage=0.0005):
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.reset()

    def reset(self):
        self.cash = self.initial_cash
        self.positions = {}  # symbol -> shares
        self.portfolio_value_history = []
        self.trades = []

    def run_backtest(self, strategy_signals, market_data, config=None):
        """
        Simulate day-by-day trading.
        :param strategy_signals: pd.DataFrame, index=date, columns=symbols, values=signals (1=buy, -1=sell, 0=hold)
        :param market_data: pd.DataFrame, index=date, columns are MultiIndex (symbol, field), e.g., ('AAPL', 'close')
        :param config: dict, optional config overrides
        :return: pd.DataFrame with portfolio value and metrics
        """
        self.reset()
        dates = strategy_signals.index
        for date in dates:
            signals = strategy_signals.loc[date]
            prices = market_data.loc[date].xs("close", level=1)
            self._execute_trades(signals, prices)
            self._update_portfolio_value(prices)
        metrics = self._calculate_metrics()
        return (
            pd.DataFrame(
                {"portfolio_value": self.portfolio_value_history, "date": dates}
            ).set_index("date"),
            metrics,
        )

    def _execute_trades(self, signals, prices):
        for symbol, signal in signals.items():
            if signal == 0 or symbol not in prices or np.isnan(prices[symbol]):
                continue
            price = prices[symbol]
            price_with_slippage = price * (1 + self.slippage * np.sign(signal))
            if signal == 1:  # Buy
                max_shares = int(
                    self.cash // (price_with_slippage * (1 + self.transaction_cost))
                )
                if max_shares > 0:
                    cost = (
                        max_shares * price_with_slippage * (1 + self.transaction_cost)
                    )
                    self.cash -= cost
                    self.positions[symbol] = self.positions.get(symbol, 0) + max_shares
                    self.trades.append(
                        {
                            "symbol": symbol,
                            "action": "buy",
                            "shares": max_shares,
                            "price": price_with_slippage,
                        }
                    )
            elif signal == -1 and self.positions.get(symbol, 0) > 0:  # Sell
                shares_to_sell = self.positions[symbol]
                proceeds = (
                    shares_to_sell * price_with_slippage * (1 - self.transaction_cost)
                )
                self.cash += proceeds
                self.positions[symbol] = 0
                self.trades.append(
                    {
                        "symbol": symbol,
                        "action": "sell",
                        "shares": shares_to_sell,
                        "price": price_with_slippage,
                    }
                )

    def _update_portfolio_value(self, prices):
        value = self.cash
        for symbol, shares in self.positions.items():
            if shares > 0 and symbol in prices and not np.isnan(prices[symbol]):
                value += shares * prices[symbol]
        self.portfolio_value_history.append(value)

    def _calculate_metrics(self):
        values = np.array(self.portfolio_value_history)
        returns = np.diff(values) / values[:-1]
        metrics = {
            "final_value": values[-1],
            "total_return": (values[-1] / values[0]) - 1,
            "max_drawdown": self._max_drawdown(values),
            "sharpe_ratio": (
                np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
                if len(returns) > 1
                else np.nan
            ),
        }
        return metrics

    def _max_drawdown(self, values):
        # Calculate maximum drawdown
        peak = values[0]
        max_drawdown = 0
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        return max_drawdown
