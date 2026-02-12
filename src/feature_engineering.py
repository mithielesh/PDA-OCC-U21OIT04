import pandas as pd


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\nBuilding Customer-Level Features...")

    customer_df = df.groupby("Customer ID").agg({
        "Revenue": "sum",
        "Invoice": "nunique",
        "Quantity": "sum",
        "InvoiceDate": ["min", "max"]
    })

    customer_df.columns = [
        "Total_Spent",
        "Total_Orders",
        "Total_Quantity",
        "First_Purchase",
        "Last_Purchase"
    ]

    customer_df = customer_df.reset_index()

    # Recency feature
    customer_df["Recency_Days"] = (
        customer_df["Last_Purchase"].max() - customer_df["Last_Purchase"]
    ).dt.days

    # Average Order Value
    customer_df["Avg_Order_Value"] = (
        customer_df["Total_Spent"] / customer_df["Total_Orders"]
    )

    # Classification Target
    median_spend = customer_df["Total_Spent"].median()
    customer_df["High_Value_Customer"] = (
        customer_df["Total_Spent"] > median_spend
    ).astype(int)

    print("Customer feature dataset shape:", customer_df.shape)

    return customer_df
