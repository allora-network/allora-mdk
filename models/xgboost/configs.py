# pylint: disable=R0801
#  Description: Configuration class for XGBoost model.
class XgboostConfig:
    """
    Configuration class for the XGBoost model.
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

    def display(self):
        """Prints out the current configuration."""
        print("XGBoost Model Configuration:")
        print(f"  params: {self.params}")
        print(f"  num_boost_round: {self.num_boost_round}")
        print(f"  early_stopping_rounds: {self.early_stopping_rounds}")
        print(f"  scaler_feature_range: {self.scaler_feature_range}")
