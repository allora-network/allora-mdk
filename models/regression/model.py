import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from models.base_model import Model
from models.regression.configs import RegressionConfig


class RegressionModel(Model):
    """Linear Regression model for regression tasks."""

    def __init__(self, model_name="regression", config=RegressionConfig(), debug=False):
        super().__init__(model_name=model_name, debug=debug)
        self.config = config  # Use the configuration class
        self.model = LinearRegression()
        self.scaler = MinMaxScaler(
            feature_range=self.config.scaler_feature_range
        )  # Initialize the scaler with configured range

    def train(self, data: pd.DataFrame):
        # Define features and target (in this case, 'close' price as target)
        x = data[["open", "high", "low", "volume"]]
        y = data["close"]

        # Normalize the features using MinMaxScaler
        x_scaled = self.scaler.fit_transform(x)

        # Train the linear regression model
        self.model.fit(x_scaled, y)

        # Save the model and scaler
        self.save()

    def inference(self, input_data: pd.DataFrame) -> pd.DataFrame:
        # Drop rows with missing values in the input data
        input_data = input_data.dropna()

        # Ensure input_data contains the necessary features
        x_test = input_data[["open", "high", "low", "volume"]]

        # Use the scaler to normalize the input data
        x_test_scaled = self.scaler.transform(x_test)

        # Predict using the trained model and convert to DataFrame
        predictions = pd.DataFrame(
            self.model.predict(x_test_scaled), columns=["prediction"]
        )

        return predictions

    def forecast(self, steps: int) -> pd.DataFrame:
        """Regression models do not forecast; return a dummy implementation."""
        return pd.DataFrame({"forecast": ["N/A"] * steps})
