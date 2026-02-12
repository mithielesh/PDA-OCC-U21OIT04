from src.data_loader import load_data
from src.data_quality import basic_profile, data_quality_score
from src.preprocessing import clean_data
from src.feature_engineering import build_customer_features
from src.model import (
    train_regression,
    train_classification,
    plot_feature_importance,
    compare_regression_models,
    compare_classification_models
)
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns



def run_pipeline(file_name: str):
    print("\n=== LOADING DATA ===")
    df = load_data(file_name)

    print("\n=== BEFORE CLEANING ===")
    basic_profile(df)
    print("Quality Score:", data_quality_score(df))

    df_clean = clean_data(df)

    print("\n=== AFTER CLEANING ===")
    basic_profile(df_clean)
    print("Quality Score:", data_quality_score(df_clean))

    customer_df = build_customer_features(df_clean)

    # Save processed dataset
    base_path = Path(__file__).resolve().parent
    processed_path = base_path / "data" / "processed" / "customer_features.csv"

    customer_df.to_csv(processed_path, index=False)
    print(f"\nSaved processed dataset to {processed_path}")

    print("\n=== CUSTOMER DATA SAMPLE ===")
    print(customer_df.head())

    print("\n=== HIGH VALUE DISTRIBUTION ===")
    print(customer_df["High_Value_Customer"].value_counts())

    images_path = base_path / "images"
    images_path.mkdir(exist_ok=True)

    # ---- Revenue Distribution ----
    plt.figure(figsize=(6, 4))
    sns.histplot(customer_df["Total_Spent"], bins=50)
    plt.title("Revenue Distribution")
    plt.xlabel("Total Spent")
    plt.ylabel("Customers")
    plt.tight_layout()
    plt.savefig(images_path / "revenue_distribution.png")
    plt.close()

    print("Saved revenue_distribution.png")

    # ---- High Value Split ----
    plt.figure(figsize=(4, 4))
    customer_df["High_Value_Customer"].value_counts().plot.pie(
        autopct="%1.1f%%"
    )
    plt.title("High Value Customer Split")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(images_path / "high_value_split.png")
    plt.close()

    print("Saved high_value_split.png")
    
    reg_model = train_regression(customer_df)
    clf_model = train_classification(customer_df)   

    reg_model, reg_features = train_regression(customer_df)
    clf_model, clf_features = train_classification(customer_df)

    print("\n=== REGRESSION FEATURE IMPORTANCE ===")
    reg_importance = plot_feature_importance(
        reg_model,
        reg_features,
        "Regression Feature Importance",
        "regression_feature_importance.png"
    )

    print("\n=== CLASSIFICATION FEATURE IMPORTANCE ===")
    clf_importance = plot_feature_importance(
        clf_model,
        clf_features,
        "Classification Feature Importance",
        "classification_feature_importance.png"
    )

    compare_regression_models(customer_df)
    compare_classification_models(customer_df)

    return customer_df


if __name__ == "__main__":
    FILE_NAME = "online_retail_II.xlsx"
    customer_df = run_pipeline(FILE_NAME)
