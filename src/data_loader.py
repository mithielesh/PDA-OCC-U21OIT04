import pandas as pd
from pathlib import Path


def load_data(file_name: str) -> pd.DataFrame:
    base_path = Path(__file__).resolve().parent.parent
    data_path = base_path / "data" / "raw" / file_name

    if not data_path.exists():
        raise FileNotFoundError(f"{file_name} not found in data/raw/")

    if file_name.endswith(".xlsx"):
        df = pd.read_excel(data_path)
    elif file_name.endswith(".csv"):
        df = pd.read_csv(data_path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .xlsx")

    return df


if __name__ == "__main__":
    FILE_NAME = "online_retail_II.xlsx"  
    df = load_data(FILE_NAME)

    print("Dataset Loaded Successfully")
    print("Shape:", df.shape)
    print("\nColumns:\n", df.columns)
    print("\nFirst 5 rows:\n", df.head())
