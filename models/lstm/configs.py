#  Description: Configuration class for LSTM model.


# pylint: disable=too-many-instance-attributes
class LstmConfig:
    """
    Configuration class for the LSTM model. This stores settings for model parameters,
    training, and data preprocessing.
    """

    def __init__(self):
        # Model architecture parameters
        self.input_size = 1  # Input size (number of features)
        self.hidden_size = 64  # Number of LSTM units per layer
        self.output_size = 1  # Output size (prediction dimension)
        self.num_layers = 2  # Number of stacked LSTM layers
        self.dropout = 0.5  # Dropout probability for regularization

        # Training parameters
        self.learning_rate = 0.0001  # Learning rate for the optimizer
        self.batch_size = 32  # Batch size for training
        self.epochs = 100  # Number of training epochs
        self.early_stopping_patience = 10  # Early stopping patience in epochs

        # Data processing
        self.validation_split = 0.2  # Proportion of data used for validation
        self.time_steps = 60  # Number of time steps used for LSTM input
        self.interval = (
            "D"  # Default to daily interval (e.g., 'D', '5M', 'H', 'W', 'M')
        )

    def display(self):
        """Prints out the current configuration."""
        print("LSTM Configuration:")
        print(f"  Input Size: {self.input_size}")
        print(f"  Hidden Size: {self.hidden_size}")
        print(f"  Output Size: {self.output_size}")
        print(f"  Num Layers: {self.num_layers}")
        print(f"  Dropout: {self.dropout}")
        print(f"  Learning Rate: {self.learning_rate}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Early Stopping Patience: {self.early_stopping_patience}")
        print(f"  Validation Split: {self.validation_split}")
        print(f"  Time Steps: {self.time_steps}")
        print(f"  Interval: {self.interval}")
