import importlib

from utils.common import snake_to_camel
from utils.model_commons import set_seed


class MetricFactory:
    """Factory class to dynamically create and manage metrics."""

    def __init__(self, seed=42):
        """Initialize the factory with a default random seed for reproducibility."""
        set_seed(seed)
        self.metrics_dir = "metrics"

    def create_metric(self, metric_name: str):
        """Dynamically import and create a metric class based on the metric_name."""
        try:
            # Dynamically construct the module path based on metric_name
            module_name = f"{self.metrics_dir}.{metric_name}.metric"

            # Import the metric module dynamically (from metric.py)
            metric_module = importlib.import_module(module_name)

            # Convert the metric_name from snake_case to CamelCase
            metric_class_name = snake_to_camel(metric_name) + "Metric"
            print(f"Metric class name: {metric_class_name}")

            # Get the metric class from the imported module
            metric_class = getattr(metric_module, metric_class_name)

            # Return an instance of the metric class
            return metric_class()
        except ModuleNotFoundError as e:
            raise ValueError(
                f"Metric '{metric_name}' not found in {self.metrics_dir}. Error: {str(e)}"
            ) from e
        except AttributeError as e:
            raise ValueError(
                f"Metric class '{metric_class_name}' not found in {metric_name}.metric. Error: {str(e)}"
            ) from e
