import pandas as pd

from metrics.base_metric import Metric
from metrics.calmar_ratio.configs import TRADING_DAYS

# Calmar Ratio measures risk-adjusted return by comparing the annualized return of an investment to its maximum drawdown.
# It helps quantify how much return you’re getting for the risk you’re taking.
#
# Formula:
# Calmar Ratio = Annualized Return / Maximum Drawdown
# Annualized Return = Mean Daily Return * 252 (assuming 252 trading days in a year)
# Maximum Drawdown = Maximum loss from a peak to a trough in the investment's value
# Note: Calmar Ratio is often used for hedge funds and other alternative investments.
#
# Reference: https://www.investopedia.com/terms/c/calmarratio.asp


class CalmarRatioMetric(Metric):
    """Calmar Ratio metric class."""

    def __init__(self, debug=False):
        super().__init__(debug=debug)

    @staticmethod
    def calculate_max_drawdown(close_prices: pd.Series) -> float:
        """Calculate the maximum drawdown from a series of closing prices."""
        # Calculate cumulative returns
        cumulative_return = (close_prices / close_prices.cummax()) - 1

        # Find the maximum drawdown (lowest point)
        max_drawdown = cumulative_return.min()
        return max_drawdown

    def calculate(self, input_data: pd.DataFrame) -> float:
        """Calculate Calmar Ratio based on daily returns and maximum drawdown."""
        # Calculate daily returns (percentage change)
        daily_returns = input_data["close"].pct_change().dropna()

        # Annualized return (assuming 252 trading days)
        annual_return = daily_returns.mean() * TRADING_DAYS

        # Calculate maximum drawdown based on closing prices
        max_drawdown = self.calculate_max_drawdown(input_data["close"])

        # Calculate Calmar Ratio: annual return / absolute value of max drawdown
        calmar_ratio = (
            annual_return / abs(max_drawdown) if max_drawdown != 0 else float("nan")
        )

        if self.debug:
            print(f"Annual return: {annual_return}")
            print(f"Max drawdown: {max_drawdown}")
            print(f"Calculated Calmar Ratio: {calmar_ratio}")

        return calmar_ratio
