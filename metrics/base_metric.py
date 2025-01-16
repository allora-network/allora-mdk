from abc import ABC, abstractmethod

import pandas as pd


class Metric(ABC):
    """Base class for all financial metrics."""

    def __init__(self, debug=False):
        self.debug = debug

    @abstractmethod
    def calculate(self, input_data: pd.DataFrame) -> float:
        """Calculate the metric based on input data."""
