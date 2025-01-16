#  Description: Configuration class for Prophet model.


# pylint: disable=too-many-instance-attributes
class ProphetConfig:
    """
    Configuration class for the Prophet model. This stores settings for model parameters,
    forecast parameters, and data preprocessing.
    """

    def __init__(self):
        # Prophet model configuration
        self.growth = "logistic"  # Options: 'linear', 'logistic'
        self.cap = None  # Optional, for logistic growth. Can be dynamically calculated if None.
        self.changepoint_prior_scale = 0.25  # Regularization strength for changepoints
        self.seasonality_mode = (
            "multiplicative"  # Options: 'additive', 'multiplicative'
        )
        self.yearly_seasonality = False  # Whether to include yearly seasonality
        self.weekly_seasonality = True  # Whether to include weekly seasonality
        self.daily_seasonality = True  # Whether to include daily seasonality

        # Forecast parameters
        self.periods = 365  # Default number of periods for future forecasts (trading days for stocks usually 252)

        # Data preprocessing
        self.remove_timezone = True  # Whether to remove timezone information from dates

    def display(self):
        """Prints out the current configuration."""
        print("Prophet Configuration:")
        print(f"  Growth: {self.growth}")
        print(f"  Changepoint Prior Scale: {self.changepoint_prior_scale}")
        print(f"  Seasonality Mode: {self.seasonality_mode}")
        print(f"  Yearly Seasonality: {self.yearly_seasonality}")
        print(f"  Weekly Seasonality: {self.weekly_seasonality}")
        print(f"  Daily Seasonality: {self.daily_seasonality}")
        print(f"  Periods: {self.periods}")
        print(f"  Remove Timezone: {self.remove_timezone}")
