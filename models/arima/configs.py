# Description: Configuration class for the ARIMA model.


# pylint: disable=too-many-instance-attributes
class ArimaConfig:
    """
    Configuration class for the ARIMA model. This stores settings for model parameters,
    grid search, and training options.
    """

    def __init__(self, use_grid_search=True, interval="D"):
        # Model parameters (p, d, q) - for manual configuration
        self.best_params = (1, 1, 0)

        # Grid Search Configuration
        self.use_grid_search = use_grid_search  # Toggle grid search
        self.p_values = [0, 1]  # Candidate values for p in ARIMA(p, d, q)
        self.d_values = [0, 1]  # Candidate values for d in ARIMA(p, d, q)
        self.q_values = [0, 1, 2]  # Candidate values for q in ARIMA(p, d, q)

        # Model training options
        self.max_iter = 100  # Maximum number of iterations for model fitting

        # Data processing
        self.frequency = "D"  # Data frequency (e.g., 'D' for daily, 'M' for monthly)
        self.interval = interval  # Time interval for resampling (e.g., '5M', 'H', 'D', 'W', 'M', 'Q', 'Y')

    def display(self):
        """Prints out the current configuration."""
        print("ARIMA Configuration:")
        print(f"  Best Params: {self.best_params}")
        print(f"  Use Grid Search: {self.use_grid_search}")
        print(f"  p Values: {self.p_values}")
        print(f"  d Values: {self.d_values}")
        print(f"  q Values: {self.q_values}")
        print(f"  Max Iterations: {self.max_iter}")
        print(f"  Data Frequency: {self.frequency}")
        print(f"  Resampling Interval: {self.interval}")
