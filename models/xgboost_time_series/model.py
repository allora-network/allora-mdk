# pylint: disable=R0801
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

from models.base_model import Model
from models.xgboost_time_series.configs import XgboostTimeSeriesConfig
from utils.model_commons import create_lag_features, split_and_scale_data


class XgboostTimeSeriesModel(Model):
    """XGBoost model for time series forecasting."""

    def __init__(
        self,
        model_name="xgboost_time_series",
        config=XgboostTimeSeriesConfig(),
        debug=False,
    ):
        super().__init__(model_name=model_name, debug=debug)
        self.config = config
        self.scaler = MinMaxScaler(
            feature_range=self.config.scaler_feature_range
        )  # Initialize the scaler with the configured range
        self.n_lags = self.config.n_lags  # Use the lag configuration from the config

    def train(self, data: pd.DataFrame):
        # Create lag features for the 'close' column
        data_with_lags = create_lag_features(data, "close", self.n_lags)

        # Define features and target
        features = data_with_lags[
            ["open", "high", "low", "volume"]
            + [f"lag_{i}" for i in range(1, self.n_lags + 1)]
        ].values
        target = data_with_lags["close"].values

        # Split data into training and validation sets Fit and transform the scaler during training
        x_train_scaled, x_val_scaled, y_train, y_val, self.scaler = (
            split_and_scale_data(
                features,
                target,
                scaler=self.scaler,
            )
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
        # Create lag features for prediction
        input_data_with_lags = create_lag_features(input_data, "close", self.n_lags)

        # Select and scale the features for prediction
        features = input_data_with_lags[
            ["open", "high", "low", "volume"]
            + [f"lag_{i}" for i in range(1, self.n_lags + 1)]
        ].values

        # Check if there are enough samples for inference
        if len(features) == 0:
            raise ValueError(
                f"Not enough data for the model. Expected at least {self.n_lags + 1} rows, but got {len(input_data)}."
            )

        features_scaled = self.scaler.transform(features)

        # Convert to XGBoost DMatrix for prediction
        dtest = xgb.DMatrix(features_scaled)

        # Predict using the trained XGBoost model
        predictions = self.model.predict(dtest)

        # Ensure the predictions have the same index as input_data_with_lags, not input_data
        predictions_df = pd.DataFrame(
            {"prediction": predictions}, index=input_data_with_lags.index
        )

        # Merge the predictions back with the original input_data index, filling NaNs for the initial rows
        result = pd.DataFrame(
            index=input_data.index
        )  # Create a DataFrame with the original input_data index
        result = result.merge(
            predictions_df, left_index=True, right_index=True, how="left"
        )  # Merge on index

        return result

    def forecast(self, steps: int) -> pd.DataFrame:
        """Dummy forecast logic, should be adapted for time series forecasting."""
        return pd.DataFrame({"forecast": [0] * steps})
