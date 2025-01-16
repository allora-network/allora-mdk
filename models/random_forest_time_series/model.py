# pylint: disable=R0801
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

from models.base_model import Model
from models.random_forest_time_series.configs import RandomForestTimeSeriesConfig
from utils.model_commons import create_lag_features, split_and_scale_data


class RandomForestTimeSeriesModel(Model):
    """Random Forest model for time series forecasting."""

    def __init__(
        self,
        model_name="random_forest_time_series",
        config=RandomForestTimeSeriesConfig(),
        debug=False,
    ):
        super().__init__(model_name=model_name, debug=debug)
        self.config = config  # Use the configuration class
        self.model = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
        )
        self.scaler = MinMaxScaler(feature_range=self.config.scaler_feature_range)
        self.n_lags = self.config.n_lags  # Use configurable number of lags

    def train(self, data: pd.DataFrame):
        # Create lag features for the 'close' column
        data_with_lags = create_lag_features(data, "close", self.n_lags)

        # Define features and target (lags as features, close as target)
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
                test_size=self.config.test_size,
                random_state=self.config.random_state,
            )
        )

        # Train the Random Forest model
        self.model.fit(x_train_scaled, y_train)

        # Evaluate the model
        val_score = self.model.score(x_val_scaled, y_val)
        print(f"Validation R^2 score: {val_score:.4f}")

        # Save the model
        self.save()

    def inference(self, input_data: pd.DataFrame) -> pd.DataFrame:
        # Ensure input_data is a DataFrame
        if not isinstance(input_data, pd.DataFrame):
            raise ValueError("Input data must be a Pandas DataFrame.")

        # Create lag features for prediction
        input_data_with_lags = create_lag_features(input_data, "close", self.n_lags)
        features = input_data_with_lags[
            ["open", "high", "low", "volume"]
            + [f"lag_{i}" for i in range(1, self.n_lags + 1)]
        ].values

        # Check if there are enough samples for inference
        if len(features) == 0:
            raise ValueError(
                f"Not enough data for the model. Expected at least {self.n_lags + 1} rows, but got {len(input_data)}."
            )

        # Use the same scaler from training to transform the input data
        features_scaled = self.scaler.transform(features)

        # Make predictions using the trained model
        predictions = self.model.predict(features_scaled)

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
        # A simple dummy forecast implementation for now
        return pd.DataFrame({"forecast": [0] * steps})
