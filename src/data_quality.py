import pandas as pd


def basic_profile(df: pd.DataFrame) -> None:
    print("\n===== BASIC INFO =====")
    print(df.info())

    print("\n===== MISSING VALUES =====")
    missing = df.isnull().sum()
    print(missing[missing > 0])

    print("\n===== DUPLICATES =====")
    print("Duplicate Rows:", df.duplicated().sum())

    print("\n===== NUMERICAL SUMMARY =====")
    print(df.describe())

    print("\n===== CATEGORICAL SUMMARY =====")
    print(df.describe(include='object'))


def data_quality_score(df: pd.DataFrame) -> float:
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()

    missing_ratio = missing_cells / total_cells
    duplicate_ratio = duplicate_rows / df.shape[0]

    score = 1 - (missing_ratio + duplicate_ratio)
    return round(score * 100, 2)
