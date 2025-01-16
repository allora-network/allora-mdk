# pylint: disable=R0801
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

from models.base_model import Model
from models.random_forest.configs import RandomForestConfig
from utils.model_commons import split_and_scale_data


class RandomForestModel(Model):
    """Random Forest model for regression tasks."""

    def __init__(
        self, model_name="random_forest", config=RandomForestConfig(), debug=False
    ):
        super().__init__(model_name=model_name, debug=debug)
        self.config = config  # Use the configuration class
        self.model = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
        )
        self.scaler = MinMaxScaler(feature_range=self.config.scaler_feature_range)

    def train(self, data: pd.DataFrame):
        # Define features and target
        features = data[["open", "high", "low", "volume"]].values
        target = data["close"].values

        # Split data into training and validation sets Fit and transform the scaler during training
        x_train_scaled, x_val_scaled, y_train, y_val, self.scaler = (
            split_and_scale_data(features, target, self.scaler)
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

        # Select features for prediction (same 4 features used during training)
        features = input_data[
            ["open", "high", "low", "volume"]
        ].values  # Convert to NumPy array

        # Use the same scaler from training to transform the input data
        features_scaled = self.scaler.transform(
            features
        )  # Make sure the scaler is fitted before calling this

        # Make predictions using the trained model
        predictions = self.model.predict(features_scaled)

        return pd.DataFrame({"prediction": predictions})

    def forecast(self, steps: int) -> pd.DataFrame:
        # A simple dummy forecast implementation for now
        return pd.DataFrame({"forecast": [0] * steps})
