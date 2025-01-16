import pandas as pd

from metrics.base_metric import Metric
from metrics.sortino_ratio.configs import RISK_FREE_RATE, TRADING_DAYS

# Sortino Ratio measures risk-adjusted return by comparing the excess return of an investment (compared to a risk-free asset) to its downside deviation.
# It helps quantify how much return you’re getting for the downside risk you’re taking.
#
# Formula:
# Sortino Ratio = (Mean Excess Return) / (Downside Deviation)
# Excess Return = Asset Return - Risk-Free Rate
# Downside Deviation = Standard Deviation of Negative Excess Returns
# Annualized Sortino Ratio = Sortino Ratio * sqrt(252)  # Assuming 252 trading days in a year
# Note: Sortino Ratio is often used for hedge funds and other alternative investments.
#
# Reference: https://www.investopedia.com/terms/s/sortinoratio.asp


class SortinoRatioMetric(Metric):
    """Sortino Ratio metric class."""

    def __init__(self, debug=False):
        super().__init__(debug=debug)
        self.risk_free_rate = RISK_FREE_RATE

    def calculate(self, input_data: pd.DataFrame) -> float:
        """Calculate Sortino Ratio based on historical returns."""
        returns = input_data["close"].pct_change().dropna()

        # Adjust for daily returns if using an annualized risk-free rate
        annual_factor = TRADING_DAYS
        excess_returns = returns - self.risk_free_rate / annual_factor

        # Calculate downside returns (only negative excess returns)
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std()

        # Calculate Sortino Ratio
        if downside_std == 0:
            return float("nan")  # Avoid division by zero
        sortino_ratio = excess_returns.mean() / downside_std

        if self.debug:
            print(f"Mean excess return: {excess_returns.mean()}")
            print(f"Downside deviation: {downside_std}")

        return sortino_ratio
