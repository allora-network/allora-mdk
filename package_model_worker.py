import argparse
import glob
import importlib.util
import os
import re
import shutil
import sys
import warnings

import pandas as pd

from utils.common import print_colored, snake_to_camel


def package_model(model_name):
    """
    Packages the model weights, scaler (if exists), inference, training scripts, API, and dependencies
    for easy integration into other projects. Also performs a test inference and training to ensure packaging is correct.

    :param model_name: Name of the model (e.g., 'lstm', 'arima').
    :param output_dir: Directory to place the packaged model.
    """

    output_dir = "package"

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
       
    # Determine model weight file (pytorch .pt or sklearn .pkl)
    if os.path.exists(f"trained_models/{model_name}/model.pt"):
        model_weight_file = f"trained_models/{model_name}/model.pt"
    elif os.path.exists(f"trained_models/{model_name}/model.pkl"):
        model_weight_file = f"trained_models/{model_name}/model.pkl"
    else:
        print_colored(
            f"Model weight not found for {model_name}, you must first train the model.",
            "error",
        )
        return

    # Check for scaler file
    scaler_file = f"trained_models/{model_name}/scaler.pkl"
    scaler_exists = os.path.exists(scaler_file)

    # Paths to model files
    model_script_dir = f"models/{model_name}/"
    model_script_file = os.path.join(model_script_dir, "model.py")

    # Additional dependencies (shared files)
    base_model_file = "models/base_model.py"

    # Check if model script file exists
    if not os.path.exists(model_script_file):
        print_colored(f"Model script not found for {model_name}", "error")
        return

    # Create directory for the specific model
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # Create trained_models directory inside packaged model folder
    model_trained_dir = os.path.join(output_dir, "trained_models", model_name)
    os.makedirs(model_trained_dir, exist_ok=True)

    # Copy model weight file to local trained_models inside packaged model
    shutil.copy(model_weight_file, model_trained_dir)

    # Copy scaler file if it exists
    if scaler_exists:
        shutil.copy(scaler_file, model_trained_dir)

    # Copy and modify import paths in the model script and its dependencies
    for py_file in glob.glob(f"{model_script_dir}/*.py"):
        copy_and_modify_imports(py_file, model_output_dir, model_name)

    # Copy and modify shared dependencies (base_model, utils, etc.)
    if os.path.exists(base_model_file):
        copy_and_modify_imports(base_model_file, model_output_dir, model_name)

    # Test inference and training to validate packaging
    test_inference(model_output_dir, model_name)
    test_training(model_output_dir, model_name)

    if scaler_exists:
        print_colored(f"Scaler also included in {model_output_dir}", "gray")
    if os.path.exists("requirements.txt"):
        print_colored(f"Requirements included in {output_dir}", "gray")
    # Print success message
    print_colored(f"Packaged {model_name} in {output_dir}", "success")


def copy_and_modify_imports(py_file, output_dir, model_name):
    """
    Copies a Python file to the output directory and modifies import paths.
    Changes lines like 'from models.base_model' to 'from packaged_models.<model_name>.base_model'.

    :param py_file: The source Python file.
    :param output_dir: The directory to copy the file to.
    :param model_name: The name of the model (e.g., 'lstm', 'arima').
    """
    with open(py_file, "r", encoding="utf-8") as file:
        content = file.read()

    # Modify 'from models...' and 'import models...' to absolute imports for the packaged model
    content = re.sub(
        r"from models\.base_model", f"from package.{model_name}.base_model", content
    )
    content = re.sub(
        r"from models\." + re.escape(model_name) + r"\.utils",
        f"from package.{model_name}.utils",
        content,
    )
    content = re.sub(
        r"from models\." + re.escape(model_name) + r"\.configs",
        f"from package.{model_name}.configs",
        content,
    )

    # Write the modified content to the output directory
    file_name = os.path.basename(py_file)
    with open(os.path.join(output_dir, file_name), "w", encoding="utf-8") as file:
        file.write(content)


def test_inference(packaged_dir, model_name):
    """
    Perform a quick inference test after packaging to ensure the model is working.

    :param packaged_dir: Directory where the model and scripts have been packaged.
    :param model_name: The name of the model being tested.
    """
    try:
        print_colored("Performing inference test...", "cyan")

        # Add the parent directory of the model to sys.path (packaged_models)
        sys.path.insert(0, os.path.dirname(os.path.dirname(packaged_dir)))

        # Suppress the specific Torch warning related to TypedStorage
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="TypedStorage is deprecated"
        )

        # Sample input data for inference (adjust as per your project)
        input_data = pd.DataFrame(
            {
                "date": pd.date_range(start="2024-09-06", periods=30, freq="D"),
                "open": [2400, 2700, 3700] * 10,
                "high": [2500, 2800, 4000] * 10,
                "low": [2000, 2000, 3500] * 10,
                "close": [1200, 2300, 3300, 2200, 2100, 3200, 1100, 2100, 2000, 2500]
                * 3,
                "volume": [1000000, 2000000, 3000000] * 10,
            }
        )

        # Load the model dynamically from the packaged directory
        model_path = os.path.join(packaged_dir, "model.py")
        spec = importlib.util.spec_from_file_location(f"{model_name}/model", model_path)
        model_module = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(model_module)  # type: ignore

        # Dynamically retrieve the model class (e.g., LstmModel)
        model_class_name = snake_to_camel(model_name) + "Model"
        model_class = getattr(model_module, model_class_name)
        model = model_class()
        model.save_dir = os.path.join(os.path.dirname(packaged_dir), "trained_models")
        print(model.save_dir)
        # Load the model (assuming it has a load method)
        model.load()

        # Perform inference
        predictions = model.inference(input_data)
        print_colored(f"Predictions:\n{predictions.head()}")

        print_colored(f"Inference test passed for {model_name}.", "success")

    # pylint: disable=broad-except
    except Exception as e:
        print_colored(f"Inference test failed for {model_name}.", "error")
        print_colored(f"Error: {e}", "error")


def test_training(packaged_dir, model_name):
    """
    Perform a quick training test after packaging to ensure the model can be trained without saving.

    :param packaged_dir: Directory where the model and scripts have been packaged.
    :param model_name: The name of the model being tested.
    """
    try:
        print_colored("Performing training test...", "cyan")

        # Add the parent directory of the model to sys.path (packaged_models)
        sys.path.insert(0, os.path.dirname(os.path.dirname(packaged_dir)))

        # Sample input data for training (larger dataset for LSTM, matching time_steps requirement)
        input_data = pd.DataFrame(
            {
                "date": pd.date_range(
                    start="2024-09-06", periods=90, freq="D"
                ),  # 90 periods for LSTM time_steps
                "open": [2400, 2700, 3700] * 30,
                "high": [2500, 2800, 4000] * 30,
                "low": [1500, 1900, 2500] * 30,
                "close": [1200, 2300, 3300, 2200, 2100, 3200, 1100, 2100, 2000, 2500]
                * 9,
                "volume": [1000000, 2000000, 3000000] * 30,
            }
        )

        # Load the model dynamically from the packaged directory
        model_path = os.path.join(packaged_dir, "model.py")
        spec = importlib.util.spec_from_file_location(f"{model_name}/model", model_path)
        model_module = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(model_module)  # type: ignore

        # Dynamically retrieve the model class (e.g., LstmModel)
        model_class_name = snake_to_camel(model_name) + "Model"
        model_class = getattr(model_module, model_class_name)
        model = model_class()
        model.save_dir = os.path.join(os.path.dirname(packaged_dir), "trained_models")
        print(model.save_dir)

        # Disable saving during training (overwrite save method)
        def dummy_save():
            print_colored(
                "Skipping model save during the package training test, to preserve the original model.",
                "warn",
            )

        model.save = dummy_save  # Override the save method

        # Train the model on the sample data (without saving)
        model.train(input_data)

        print_colored(f"Training test passed for {model_name}.", "success")

    # pylint: disable=broad-except
    except Exception as e:
        print_colored(f"Training test failed for {model_name}.", "error")
        print_colored(f"Error: {e}", "error")


if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(
        description="Package a trained model for worker integration."
    )
    parser.add_argument(
        "model_name", type=str, help="Name of the model to package (e.g., lstm, arima)"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the package_model function with the parsed arguments
    package_model(args.model_name)
