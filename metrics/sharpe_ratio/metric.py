import pandas as pd

from metrics.base_metric import Metric
from metrics.sharpe_ratio.configs import RISK_FREE_RATE, TRADING_DAYS, WINDOW_SIZE

# Sharpe Ratio measures risk-adjusted return by comparing the excess return of an investment (compared to a risk-free asset) to its standard deviation (or volatility).
# It helps quantify how much return you’re getting for the risk you’re taking.
#
# Formula:
# Sharpe Ratio = (Mean Excess Return) / (Standard Deviation of Excess Return)
# Excess Return = Asset Return - Risk-Free Rate
# Annualized Sharpe Ratio = Sharpe Ratio * sqrt(252)  # Assuming 252 trading days in a year
# Note: Sharpe Ratio is often used for stocks and other traditional investments.
#
# Reference: https://www.investopedia.com/terms/s/sharperatio.asp


class SharpeRatioMetric(Metric):
    """Sharpe Ratio metric class."""

    def __init__(self, debug=False):
        super().__init__(debug=debug)
        self.risk_free_rate = RISK_FREE_RATE
        self.window_size = WINDOW_SIZE

    def calculate(self, input_data: pd.DataFrame) -> float:
        """Calculate Sharpe Ratio based on historical returns."""
        returns = input_data["close"].pct_change().dropna()

        # Adjust the risk-free rate for daily returns (assuming an annual risk-free rate)
        annual_factor = TRADING_DAYS
        excess_returns = returns - self.risk_free_rate / annual_factor

        # Calculate Sharpe Ratio: mean of excess returns / standard deviation of excess returns
        excess_mean = excess_returns.mean()
        excess_std = excess_returns.std()

        # Handle edge case where standard deviation is zero
        if excess_std == 0:
            return float("nan")

        sharpe_ratio = excess_mean / excess_std

        if self.debug:
            print(f"Mean excess return: {excess_mean}")
            print(f"Standard deviation of excess returns: {excess_std}")
            print(f"Calculated Sharpe Ratio: {sharpe_ratio}")

        return sharpe_ratio

    def rolling_sharpe(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling Sharpe Ratio over a given window (default {TRADING_DAYS} trading days)."""
        # Calculate returns and excess returns
        input_data["returns"] = input_data["close"].pct_change()
        input_data["excess_returns"] = (
            input_data["returns"] - self.risk_free_rate / TRADING_DAYS
        )  # Adjust for daily returns

        # Apply rolling mean and standard deviation over the specified window
        rolling_mean = input_data["excess_returns"].rolling(self.window_size).mean()
        rolling_std = input_data["excess_returns"].rolling(self.window_size).std()

        # Calculate rolling Sharpe Ratio: handle cases where std is 0
        input_data["rolling_sharpe"] = rolling_mean / rolling_std
        input_data["rolling_sharpe"] = input_data["rolling_sharpe"].replace(
            [float("inf"), -float("inf")], float("nan")
        )

        if self.debug:
            print(
                input_data[["date", "rolling_sharpe"]].tail()
            )  # Print last few rows for inspection

        return input_data[["date", "rolling_sharpe"]]
