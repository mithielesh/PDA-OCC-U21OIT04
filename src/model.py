from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def train_regression(customer_df):
    print("\n=== REGRESSION MODEL ===")

    X = customer_df.drop(
        columns=["Customer ID", "Total_Spent", "High_Value_Customer",
                 "First_Purchase", "Last_Purchase"]
    )
    y = customer_df["Total_Spent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))

    return model, X.columns


def train_classification(customer_df):
    print("\n=== CLASSIFICATION MODEL ===")

    X = customer_df.drop(
        columns=["Customer ID", "High_Value_Customer",
                 "Total_Spent", "First_Purchase", "Last_Purchase"]
    )
    y = customer_df["High_Value_Customer"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)

    print("Accuracy:", round(acc, 4))
    print("\nClassification Report:\n", classification_report(y_test, predictions))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))

    return model, X.columns

def plot_feature_importance(model, feature_names, title, file_name):
    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    # Plot
    plt.figure()
    plt.bar(importance_df["Feature"], importance_df["Importance"])
    plt.xticks(rotation=45)
    plt.title(title)
    plt.tight_layout()

    # Save image
    base_path = Path(__file__).resolve().parent.parent
    image_path = base_path / "images" / file_name

    plt.savefig(image_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved feature importance plot to {image_path}")

    return importance_df

def compare_regression_models(customer_df):
    print("\n=== REGRESSION MODEL COMPARISON ===")

    X = customer_df.drop(
        columns=["Customer ID", "Total_Spent", "High_Value_Customer",
                 "First_Purchase", "Last_Purchase"]
    )
    y = customer_df["Total_Spent"]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42)
    }

    for name, model in models.items():
        scores = cross_val_score(
            model, X, y,
            cv=5,
            scoring="r2"
        )
        print(f"{name} | Mean R2: {round(scores.mean(), 4)} | Std: {round(scores.std(), 4)}")

def compare_classification_models(customer_df):
    print("\n=== CLASSIFICATION MODEL COMPARISON ===")

    X = customer_df.drop(
        columns=["Customer ID", "High_Value_Customer",
                 "Total_Spent", "First_Purchase", "Last_Purchase"]
    )
    y = customer_df["High_Value_Customer"]

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    for name, model in models.items():
        scores = cross_val_score(
            model, X, y,
            cv=5,
            scoring="accuracy"
        )
        print(f"{name} | Mean Accuracy: {round(scores.mean(), 4)} | Std: {round(scores.std(), 4)}")
