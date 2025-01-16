import argparse

import pandas as pd


class CSVStandardizer:
    """
    A class for standardizing CSV files by cleaning and fixing common issues,
    such as inconsistent separators, date parsing, and handling missing values.
    """

    @staticmethod
    def fix_csv(
        file_path: str, output_path: str, date_column: str = None  # type: ignore
    ) -> pd.DataFrame:
        """
        Load and clean a poorly structured CSV and output it in a standard format.

        :param file_path: Path to the input CSV file.
        :param output_path: Path where the cleaned CSV will be saved.
        :param date_column: Name of the date column to parse as datetime.
        :return: Cleaned pandas DataFrame.
        """
        try:
            # Try to load the CSV with semicolon separator
            data = pd.read_csv(file_path, sep=";", engine="python")

            # If a date column is specified, parse it
            if date_column:
                if date_column in data.columns:
                    data[date_column] = pd.to_datetime(
                        data[date_column], errors="coerce"
                    )  # Parse dates and handle errors
                else:
                    raise ValueError(
                        f"Date column '{date_column}' not found in the data."
                    )

            # Remove leading/trailing spaces from column names
            data.columns = data.columns.str.strip()

            # Handle missing values by dropping rows with NaN values
            data = data.dropna()

            # Save the cleaned CSV to the specified output path
            data.to_csv(output_path, index=False)
            print(f"CSV file has been cleaned and saved to {output_path}")
            return data

        except pd.errors.ParserError as pe:
            print(f"Parser error while processing the CSV file: {pe}")
            return None  # type: ignore
        except FileNotFoundError as fnfe:
            print(f"File not found: {fnfe}")
            return None  # type: ignore
        except ValueError as ve:
            print(f"Value error: {ve}")
            return None  # type: ignore
        # pylint: disable=broad-except
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None  # type: ignore


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Clean a CSV file and output it in a standard format."
    )
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("output_file", help="Path to save the cleaned CSV file")
    parser.add_argument(
        "--date_column",
        help="Name of the date column to parse as datetime",
        default=None,
    )

    args = parser.parse_args()

    # Run the CSV cleaning process
    csv_cleaner = CSVStandardizer()
    cleaned_data = csv_cleaner.fix_csv(
        args.input_file, args.output_file, date_column=args.date_column
    )
