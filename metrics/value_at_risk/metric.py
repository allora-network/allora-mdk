import pandas as pd

from metrics.base_metric import Metric

# Value at Risk (VaR) metric measures the maximum potential loss of an investment at a given confidence level.
# It is a popular risk metric used in the financial industry.
# The metric is calculated as follows:
# 1. Calculate the returns.
# 2. Calculate the VaR at the given confidence level.
# VaR is the value below which a given percentage of returns fall.
# The metric is useful for risk management and is often used in the financial industry.
#
# Reference: https://www.investopedia.com/terms/v/var.asp


class ValueAtRiskMetric(Metric):
    """Value at Risk (VaR) metric class."""

    def __init__(self, confidence_level=0.05, debug=False):
        super().__init__(debug=debug)
        self.confidence_level = confidence_level

    def calculate(self, input_data: pd.DataFrame) -> float:
        """Calculate VaR (Value at Risk) at the given confidence level."""
        returns = input_data["close"].pct_change().dropna()
        var = returns.quantile(self.confidence_level)

        if self.debug:
            print(
                f"Value at Risk (VaR) at {self.confidence_level * 100}% confidence level: {var}"
            )

        return var
