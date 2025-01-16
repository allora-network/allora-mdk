import pandas as pd

from metrics.base_metric import Metric

# Maximum Drawdown (MDD) metric measures the maximum loss from a peak to a trough of a portfolio.
# It helps quantify the risk of an investment strategy.
#
# Formula:
# Maximum Drawdown = (Peak Value - Trough Value) / Peak Value
# Peak Value: Highest cumulative return value
# Trough Value: Lowest cumulative return value
#
# Reference: https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp


class MaximumDrawdownMetric(Metric):
    """Maximum Drawdown (MDD) metric class."""

    def calculate(self, input_data: pd.DataFrame) -> float:
        """Calculate Maximum Drawdown based on historical returns."""
        # Calculate cumulative returns
        cumulative_returns = (1 + input_data["close"].pct_change()).cumprod()

        # Calculate the drawdown
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak

        # Maximum drawdown
        max_drawdown = drawdown.min()

        if self.debug:
            print(f"Maximum drawdown: {max_drawdown}")

        return max_drawdown
