""" Common functions for working with models. """

import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def set_seed(seed):
    """Set seed for reproducibility across different libraries."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # If using CUDA, you may want to set a CUDA seed as well
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_lag_features(
    data: pd.DataFrame, target_col: str, n_lags: int
) -> pd.DataFrame:
    """Create lag features for the target column."""
    df = data.copy()
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    return df.dropna()


def split_and_scale_data(features, target, scaler=None, test_size=0.2, random_state=42):
    """
    Split the data and scale the features. Uses the provided scaler if available,
    otherwise creates a new one.

    :param features: Features (X)
    :param target: Target (y)
    :param scaler: Optional. Use this scaler if provided.
    :param test_size: The proportion of data to be used as validation set.
    :param random_state: Random state for reproducibility.
    :return: Scaled training and validation data, target, and the scaler used.
    """
    # Split data
    x_train, x_val, y_train, y_val = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )

    # If a scaler is provided, use it, otherwise create a new one
    if scaler is None:
        scaler = MinMaxScaler()

    # Scale the features
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    return x_train_scaled, x_val_scaled, y_train, y_val, scaler
