#  Description: Configuration class for Random Forest Time Series model.


class RandomForestTimeSeriesConfig:
    """
    Configuration class for the RandomForestTimeSeries model.
    This stores hyperparameters for the model and other settings for data preprocessing.
    """

    def __init__(self):
        # RandomForest hyperparameters
        self.n_estimators = 100  # Number of trees in the forest
        self.max_depth = None  # Maximum depth of the tree
        self.random_state = 42  # Seed for reproducibility
        self.test_size = 0.2  # Proportion of data to use for validation
        self.n_lags = 5  # Number of lag features

        # Data preprocessing
        self.scaler_feature_range = (0, 1)  # Range for the MinMaxScaler

    def display(self):
        """Prints out the current configuration."""
        print("RandomForestTimeSeries Configuration:")
        print(f"  n_estimators: {self.n_estimators}")
        print(f"  max_depth: {self.max_depth}")
        print(f"  random_state: {self.random_state}")
        print(f"  test_size: {self.test_size}")
        print(f"  n_lags: {self.n_lags}")
        print(f"  scaler_feature_range: {self.scaler_feature_range}")
