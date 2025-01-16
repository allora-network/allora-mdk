#  Description: Configuration class for Random Forest model.


class RandomForestConfig:
    """
    Configuration class for the Random Forest model.
    This stores hyperparameters for the model and other settings for data preprocessing.
    """

    def __init__(self):
        # RandomForest hyperparameters
        self.n_estimators = 100  # Number of trees in the forest
        self.max_depth = None  # Maximum depth of the tree
        self.random_state = 42  # Seed for reproducibility
        self.test_size = 0.2  # Proportion of data to use for validation

        # Data preprocessing
        self.scaler_feature_range = (0, 1)  # Range for the MinMaxScaler

    def display(self):
        """Prints out the current configuration."""
        print("RandomForest Configuration:")
        print(f"  n_estimators: {self.n_estimators}")
        print(f"  max_depth: {self.max_depth}")
        print(f"  random_state: {self.random_state}")
        print(f"  test_size: {self.test_size}")
        print(f"  scaler_feature_range: {self.scaler_feature_range}")
