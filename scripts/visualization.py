# scripts/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


def plot_feature_importance(model, feature_names, output_folder="outputs/model_performance"):
    """Plot feature importance."""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get feature importances
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title("Feature Importance")
    plt.savefig(f"{output_folder}/feature_importance.png")
    plt.close()