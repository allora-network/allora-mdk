import pandas as pd

from metrics.base_metric import Metric

# Expected Shortfall (ES) metric measures the expected loss in the worst case scenarios.
# It is the average of the losses that are greater than the Value at Risk (VaR).
# The metric is useful for risk management and is often used in the financial industry.
# The metric is calculated as follows:
# 1. Calculate the VaR (Value at Risk) at the given confidence level.
# 2. Filter for returns below VaR (losses).
# 3. Calculate the Expected Shortfall as the mean of the returns below VaR.
#
# Reference: https://www.investopedia.com/terms/e/expected-shortfall.aspå


class ExpectedShortfallMetric(Metric):
    """Expected Shortfall (ES) metric class."""

    def __init__(self, confidence_level=0.05, debug=False):
        super().__init__(debug=debug)
        self.confidence_level = confidence_level

    def calculate(self, input_data: pd.DataFrame) -> float:
        """Calculate ES (Expected Shortfall) at the given confidence level."""
        returns = input_data["close"].pct_change().dropna()

        # Calculate VaR (Value at Risk) at the given confidence level
        var = returns.quantile(self.confidence_level)

        # Filter for returns below VaR (losses)
        losses_below_var = returns[returns < var]

        if self.debug:
            print("Returns:")
            print(returns)
            print(f"Value at Risk (VaR) at {self.confidence_level * 100}%: {var}")
            print("Losses below VaR:")
            print(losses_below_var)

        # Check if there are any returns below VaR
        if len(losses_below_var) == 0:
            print("No losses below VaR, returning var for Expected Shortfall.")
            return var

        # Calculate Expected Shortfall as the mean of the returns below VaR
        es = losses_below_var.mean()

        if self.debug:
            print(f"Expected Shortfall (ES): {es}")

        return es
