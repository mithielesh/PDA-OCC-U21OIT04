import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\nStarting Cleaning Process...")

    initial_shape = df.shape

    # Remove duplicates
    df = df.drop_duplicates()

    # Drop missing critical values
    df = df.dropna(subset=["Customer ID", "Description"])

    # Remove invalid transactions
    df = df[df["Quantity"] > 0]
    df = df[df["Price"] > 0]

    # Convert Customer ID to integer
    df["Customer ID"] = df["Customer ID"].astype(int)

    # Create Revenue column
    df["Revenue"] = df["Quantity"] * df["Price"]

    print(f"Initial Shape: {initial_shape}")
    print(f"Final Shape after Cleaning: {df.shape}")

    return df
