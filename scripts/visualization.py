# scripts/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


def generate_plots(model, data):
    """Generate and save visualization plots."""
    output_dir = "output/eda_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Example: Feature Importance Plot (For Tree-based models like XGBoost)
    if hasattr(model, "feature_importances_"):
        plt.figure(figsize=(10, 6))
        sns.barplot(x=data.columns, y=model.feature_importances_)
        plt.xticks(rotation=45)
        plt.title("Feature Importance")
        plt.savefig(f"{output_dir}/feature_importance.png")
        plt.close()

    # Example: Histogram of a feature
    plt.figure(figsize=(8, 5))
    sns.histplot(data.iloc[:, 0], kde=True, bins=30)
    plt.title("Distribution of Feature 1")
    plt.savefig(f"{output_dir}/feature_distribution.png")
    plt.close()

    print("Plots saved in output/eda_plots/")