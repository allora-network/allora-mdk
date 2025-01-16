#  Description: Configuration class for Regression model.
class RegressionConfig:
    """
    Configuration class for the Regression model.
    This stores hyperparameters for the model and other settings for data preprocessing.
    """

    def __init__(self):
        # Data preprocessing
        self.scaler_feature_range = (0, 1)  # Range for the MinMaxScaler

    def display(self):
        """Prints out the current configuration."""
        print("Regression Model Configuration:")
        print(f"  scaler_feature_range: {self.scaler_feature_range}")
