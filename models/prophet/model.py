import pandas as pd
from prophet import Prophet

from models.base_model import Model
from models.prophet.configs import ProphetConfig


class ProphetModel(Model):
    """Prophet model for time series forecasting"""

    def __init__(self, model_name="prophet", config=ProphetConfig(), debug=False):
        super().__init__(model_name=model_name, debug=debug)
        self.config = config  # Use the configuration class
        self.model = Prophet(
            growth=self.config.growth,
            changepoint_prior_scale=self.config.changepoint_prior_scale,
            yearly_seasonality=self.config.yearly_seasonality,  # type: ignore
            weekly_seasonality=self.config.weekly_seasonality,  # type: ignore
            daily_seasonality=self.config.daily_seasonality,  # type: ignore
            seasonality_mode=self.config.seasonality_mode,
        )

    def train(self, data: pd.DataFrame):
        df = data[["date", "close"]].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Drop rows with NaN values in 'date' or 'close'
        df = df.dropna(subset=["date", "close"])

        if self.config.remove_timezone:
            df["date"] = df["date"].dt.tz_localize(None)

        df = df.rename(columns={"date": "ds", "close": "y"})

        # Handle logistic growth: Set 'cap' and 'floor' values
        if self.config.growth == "logistic":
            max_y = df["y"].max()
            df["cap"] = max_y * 1.1  # Set cap to 10% above max value
            df["floor"] = 0  # Set a floor to prevent negative growth

        if self.debug:
            # Print data to check for NaNs or extreme values
            print(df.isna().sum())
            print(df.describe())

        # Fit model
        self.model.fit(df, iter=20000)

        self.save()

    def inference(self, input_data: pd.DataFrame) -> pd.DataFrame:
        if self.debug:
            print("Input Data for ProphetModel before predictions:")
            print(input_data)

        # Ensure the 'date' column exists in the input data before proceeding
        if "date" not in input_data.columns:
            raise KeyError(
                "The input_data must contain a 'date' column for Prophet predictions."
            )

        # Rename 'date' to 'ds' for Prophet
        future = input_data[["date"]].rename(columns={"date": "ds"}).copy()

        if self.config.remove_timezone:
            future["ds"] = pd.to_datetime(future["ds"]).dt.tz_localize(None)

        # Handle logistic growth: Ensure 'cap' and 'floor' columns are present
        if self.config.growth == "logistic":
            if "cap" not in input_data.columns:
                # Dynamically set 'cap' if missing (assuming similar logic as training)
                future["cap"] = (
                    input_data["close"].max() * 1.1
                )  # Example logic for setting 'cap'

            if "floor" not in input_data.columns:
                # Set the 'floor' if it's missing (e.g., default to 0)
                future["floor"] = 0

        if self.debug:
            print("Future DataFrame for ProphetModel (after renaming):")
            print(future)

        # Perform the forecast using Prophet
        forecast = self.model.predict(future, False)
        if self.debug:
            print("Forecast Output:")
            print(forecast[["ds", "yhat"]])

        return pd.DataFrame({"date": forecast["ds"], "prediction": forecast["yhat"]})

    def forecast(self, steps: int) -> pd.DataFrame:
        future_dates = self.model.make_future_dataframe(periods=steps)
        forecast = self.model.predict(future_dates)
        return forecast[["ds", "yhat"]].tail(steps)
