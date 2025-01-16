import pandas as pd

from metrics.base_metric import Metric
from metrics.volatility.configs import TRADING_DAYS

# Volatility metric measures the standard deviation of returns.
# It is a popular risk metric used in the financial industry.
# The metric is calculated as follows:
# 1. Calculate the returns.
# 2. Calculate the standard deviation of returns.
# The metric is useful for risk management and is often used in the financial industry.
#
# Reference: https://www.investopedia.com/terms/v/volatility.asp


class VolatilityMetric(Metric):
    """Volatility metric class."""

    def calculate(self, input_data: pd.DataFrame) -> float:
        """Calculate Volatility (standard deviation of returns)."""
        returns = input_data["close"].pct_change().dropna()
        volatility = returns.std() * (TRADING_DAYS**0.5)  # Annualized volatility

        if self.debug:
            print(f"Volatility: {volatility}")

        return volatility
