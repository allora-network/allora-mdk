import pandas as pd


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by validating, cleaning, and normalizing it.

    :param data: The raw input data
    :return: Cleaned and validated data ready for training
    :raises: ValueError if validation fails
    """
    required_columns = ["open", "high", "low", "volume", "close"]

    # Check if all required columns are present
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    # Drop rows with missing values in required columns
    data = data.dropna(subset=required_columns)
    # print sample
    print(data.head())

    # Ensure all required columns have numeric data
    for col in required_columns:
        if not pd.api.types.is_numeric_dtype(data[col]):
            raise ValueError(f"Column {col} contains non-numeric values.")

    # Optionally: You can normalize or scale the data here if needed
    # Example: data[required_columns] = (data[required_columns] - data[required_columns].mean()) / data[required_columns].std()

    print("Data validation and preprocessing completed successfully.")
    return data
