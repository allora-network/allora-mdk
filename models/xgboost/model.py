# pylint: disable=R0801
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

from models.base_model import Model
from models.xgboost.configs import XgboostConfig
from utils.model_commons import split_and_scale_data


class XgboostModel(Model):
    """XGBoost model for regression tasks."""

    def __init__(self, model_name="xgboost", config=XgboostConfig(), debug=False):
        super().__init__(model_name=model_name, debug=debug)
        self.config = config
        self.scaler = MinMaxScaler(
            feature_range=self.config.scaler_feature_range
        )  # Initialize the scaler with the configured range

    def train(self, data: pd.DataFrame):
        # Define features and target
        features = data[["open", "high", "low", "volume"]].values
        target = data["close"].values

        # Split data into training and validation sets Fit and transform the scaler during training
        x_train_scaled, x_val_scaled, y_train, y_val, self.scaler = (
            split_and_scale_data(features, target, scaler=self.scaler)
        )

        # Convert data to XGBoost DMatrix
        dtrain = xgb.DMatrix(x_train_scaled, label=y_train)
        dval = xgb.DMatrix(x_val_scaled, label=y_val)

        # Specify validation set for early stopping
        evals = [(dval, "eval"), (dtrain, "train")]

        # Train the XGBoost model using the 'train' API
        self.model = xgb.train(
            self.config.params,
            dtrain,
            num_boost_round=self.config.num_boost_round,
            evals=evals,
            early_stopping_rounds=self.config.early_stopping_rounds,
            verbose_eval=True,
        )

        # Save the model after training
        self.save()

    def inference(self, input_data: pd.DataFrame) -> pd.DataFrame:
        # Select and scale the features for prediction
        features = input_data[["open", "high", "low", "volume"]].values
        features_scaled = self.scaler.transform(features)

        # Convert to XGBoost DMatrix for prediction
        dtest = xgb.DMatrix(features_scaled)

        # Predict using the trained XGBoost model
        predictions = self.model.predict(dtest)

        return pd.DataFrame({"prediction": predictions})

    def forecast(self, steps: int) -> pd.DataFrame:
        # Dummy forecast logic for now
        return pd.DataFrame({"forecast": [0] * steps})
