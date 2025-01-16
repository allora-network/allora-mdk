import pandas as pd

from metrics.base_metric import Metric
from metrics.cagr.configs import TRADING_DAYS

# CAGR means Compound Annual Growth Rate and is a measure of the mean annual growth rate of an investment over a specified time period longer than one year.
# The formula for CAGR is:
# CAGR = (EV / BV)^(1/n) - 1
# where:
# EV = Ending Value
# BV = Beginning Value
# n = Number of periods
# The CAGR formula is derived from the compound interest formula.
# The CAGR metric is useful for comparing investments with different time periods.
# For example, if you have two investments with different time periods, you can use CAGR to compare their annual growth rates.
# The CAGR metric is often used in finance and investing to analyze the performance of investments over time.
# Reference: https://www.investopedia.com/terms/c/cagr.asp


class CagrMetric(Metric):
    """CAGR metric class."""

    def __init__(self, use_trading_days=True, debug=False):
        super().__init__(debug=debug)
        self.use_trading_days = (
            use_trading_days  # Option to use trading days or calendar days
        )

    def calculate(self, input_data: pd.DataFrame) -> float:
        """Calculate CAGR (Compound Annual Growth Rate)."""
        start_value = input_data["close"].iloc[0]
        end_value = input_data["close"].iloc[-1]

        # Determine the number of periods (years)
        if self.use_trading_days:
            periods = len(input_data) / TRADING_DAYS
        else:
            total_days = (input_data["date"].iloc[-1] - input_data["date"].iloc[0]).days
            periods = total_days / 365

        # Calculate CAGR
        cagr = (end_value / start_value) ** (1 / periods) - 1

        if self.debug:
            print(
                f"Start Value: {start_value}, End Value: {end_value}, Periods: {periods}"
            )
            print(f"CAGR: {cagr}")

        return cagr
