# pylint: disable=R0801
#  Description: Configuration class for XGBoost Time Series model.
class XgboostTimeSeriesConfig:
    """
    Configuration class for the XGBoost Time Series model.
    Stores hyperparameters for the model and settings for data preprocessing.
    """

    def __init__(self):
        # XGBoost hyperparameters
        self.params = {
            "objective": "reg:squarederror",  # Regression objective
            "eval_metric": "rmse",  # Evaluation metric
        }
        self.num_boost_round = 100
        self.early_stopping_rounds = 10

        # Data preprocessing
        self.scaler_feature_range = (0, 1)  # Range for MinMaxScaler
        self.n_lags = 5  # Default number of lag features for time series

    def display(self):
        """Prints out the current configuration."""
        print("XGBoost Time Series Model Configuration:")
        print(f"  params: {self.params}")
        print(f"  num_boost_round: {self.num_boost_round}")
        print(f"  early_stopping_rounds: {self.early_stopping_rounds}")
        print(f"  scaler_feature_range: {self.scaler_feature_range}")
        print(f"  n_lags: {self.n_lags}")
