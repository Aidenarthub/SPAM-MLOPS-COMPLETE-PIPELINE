import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(log_dir / "data_ingestion.log")
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.debug("Data loaded from %s", data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred while loading the data: %s", e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
        df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
        logger.debug("Data preprocessing completed")
        return df
    except KeyError as e:
        logger.error("Missing column in the dataframe: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error during preprocessing: %s", e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, output_dir: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = Path(output_dir) / "raw"
        raw_data_path.mkdir(parents=True, exist_ok=True)
        train_data.to_csv(raw_data_path / "train.csv", index=False)
        test_data.to_csv(raw_data_path / "test.csv", index=False)
        logger.debug("Train and Test data saved to %s", raw_data_path.as_posix())
    except Exception as e:
        logger.error("Unexpected error occurred while saving the data: %s", e)
        raise

def main():
    try:
        test_size = 0.2
        data_url = "https://raw.githubusercontent.com/vikashishere/YT-MLOPS-Complete-ML-Pipeline/7b6959709ce419648103993fb8c3faf847e2db72/experiments/spam.csv"
        df = load_data(data_url=data_url)
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)
        save_data(train_data, test_data, output_dir="./data")
    except Exception as e:
        logger.error("Failed to complete the data ingestion process: %s", e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()