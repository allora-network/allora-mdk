#  Description: Configuration class for Regression Time Series model.
class RegressionTimeSeriesConfig:
    """
    Configuration class for the Regression Time Series model.
    Stores hyperparameters for the model and other settings for data preprocessing.
    """

    def __init__(self):
        # Number of lag features to create
        self.n_lags = 5

        # Data preprocessing
        self.scaler_feature_range = (0, 1)  # Range for the MinMaxScaler

    def display(self):
        """Prints out the current configuration."""
        print("Regression Time Series Model Configuration:")
        print(f"  n_lags: {self.n_lags}")
        print(f"  scaler_feature_range: {self.scaler_feature_range}")
