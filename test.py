import pandas as pd

# pylint: disable=no-name-in-module
from configs import metrics, models
from metrics.metric_factory import MetricFactory
from models.model_factory import ModelFactory
from utils.common import print_colored

# Simulate some input data for testing/prediction
input_data = pd.DataFrame(
    {
        "date": pd.date_range(start="2024-09-06", periods=30, freq="D"),
        "open": [2400, 2700, 3700] * 10,
        "high": [2500, 2800, 4000] * 10,
        "low": [1500, 1900, 2500] * 10,
        # Introduce some volatility in the 'close' prices
        "close": [1200, 2300, 3300, 2200, 2100, 3200, 1100, 2100, 2000, 2500] * 3,
        "volume": [1000000, 2000000, 3000000] * 10,
    }
)


def test_models():
    # List of model types that you want to test

    # Initialize ModelFactory
    factory = ModelFactory()

    # Loop through each model type and test predictions
    for model_name in models:

        try:
            print(f"Loading {model_name} model...")
            model = factory.create_model(model_name)
        # pylint: disable=broad-except
        except Exception as e:
            print(f"Error: Model {model_name} not found. Exception: {e}")
            continue

        model.load()

        try:
            # Call model.inference() to get predictions
            predictions = model.inference(input_data)
            print(f"Making predictions with the {model_name} model...")

            if model_name in ("prophet", "arima", "lstm"):
                print(f"{model_name.replace('_',' ').capitalize()} Model Predictions:")
                print(predictions)
            else:
                # Standardize predictions: convert DataFrame to NumPy array if necessary, and flatten
                if isinstance(predictions, pd.DataFrame):
                    predictions = (
                        predictions.values
                    )  # Convert DataFrame to NumPy array if it's a DataFrame

                if predictions.ndim == 2:
                    predictions = predictions.ravel()  # Flatten if it's a 2D array

                # Output predictions
                print(f"{model_name.capitalize()} Model Predictions:")
                print(pd.DataFrame({"prediction": predictions}, index=input_data.index))

        # pylint: disable=broad-except
        except Exception as e:
            print_colored(
                f"Error: Model {model_name} not found. Exception: {e}", "error"
            )
            continue


def test_metrics():
    # Initialize MetricFactory
    factory = MetricFactory()

    # Loop through each metric type and test calculations
    for metric_name in metrics:
        print(f"Loading {metric_name} metric...")
        metric = factory.create_metric(metric_name)

        print(f"Calculating {metric_name} metric...")

        # Call metric.calculate() to get metric value
        value = metric.calculate(input_data)

        # Output metric value
        print(f"{metric_name.capitalize()} Value:")
        print(value)


def main():
    print("Do you want to test models, metrics, or both?")
    print("1. Models")
    print("2. Metrics")
    print("3. Both")

    selection = input("Enter your choice (1/2/3): ").strip()

    if selection == "1":
        test_models()
    elif selection == "2":
        test_metrics()
    elif selection == "3":
        test_models()
        test_metrics()
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
