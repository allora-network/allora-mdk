import pandas as pd

from metrics.base_metric import Metric

# MDD Duration metric measures the length of time the portfolio remained in drawdown.
# It helps quantify the risk of an investment strategy.
#
# Formula:
# MDD Duration = Length of time the portfolio remained in drawdown
#
# Reference: https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp


class MddDurationMetric(Metric):
    """MDD Duration metric class."""

    def calculate(self, input_data: pd.DataFrame) -> int:
        """Calculate MDD Duration (the length of time the portfolio remained in drawdown)."""
        # Calculate cumulative returns
        cumulative_returns = (1 + input_data["close"].pct_change()).cumprod()

        # Calculate drawdown
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak

        # Find periods in drawdown
        drawdown_periods = (drawdown < 0).astype(int)

        # Calculate the longest drawdown duration
        mdd_duration = (
            drawdown_periods
            * (
                drawdown_periods.groupby(
                    (drawdown_periods != drawdown_periods.shift()).cumsum()
                ).cumcount()
                + 1
            )
        ).max()

        if self.debug:
            print(f"MDD Duration: {mdd_duration}")

        return mdd_duration
