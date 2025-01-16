""" Common utility functions used across the package. """

import importlib.util
import os


def snake_to_camel(snake_str):
    """Convert snake_case to CamelCase (PascalCase)."""
    components = snake_str.split("_")
    return "".join(x.capitalize() for x in components)


def print_colored(message, color=None):
    """Print a message in the specified color."""
    # ANSI escape sequences for colors
    colors = {
        "gray": "\033[90m",
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m",
    }

    # Commonly used levels that map to specific colors
    level_colors = {
        "info": colors["blue"],
        "warn": colors["yellow"],
        "error": colors["red"],
        "success": colors["green"],
    }

    # If the color or level key is recognized, apply the color, else use no color (reset)
    color_code = colors.get(color) or level_colors.get(color, colors["reset"])  # type: ignore

    # Print the message with the corresponding color
    print(f"{color_code}{message}{colors['reset']}")


def load_model_from_config(active_model):
    """
    Load the model dynamically from the given model directory.
    """
    if active_model is None:
        raise FileNotFoundError("No model specified in the environment variable MODEL.")

    model_file = os.path.join(f"package/{active_model}", "model.py")
    if os.path.exists(model_file):
        spec = importlib.util.spec_from_file_location(active_model, model_file)
        model_module = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(model_module)  # type: ignore
        model_class_name = snake_to_camel(active_model) + "Model"
        model_class = getattr(model_module, model_class_name)
        model = model_class()
        model.save_dir = "package/trained_models"
        return model

    raise FileNotFoundError(f"Model {active_model} not found.")
