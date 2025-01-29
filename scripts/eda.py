# scripts/eda.py
import matplotlib.pyplot as plt
import seaborn as sns
import os


def perform_eda(data, output_folder="outputs/eda_plots"):
    """Perform EDA and save plots."""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Display basic info
    print("Data Info:")
    print(data.info())

    print("\nMissing Values:")
    print(data.isnull().sum())

    print("\nDescriptive Statistics:")
    print(data.describe())

    # Visualize distributions
    for column in data.select_dtypes(include=['float64', 'int64']).columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(data[column], kde=True)
        plt.title(f"Distribution of {column}")
        plt.savefig(f"{output_folder}/{column}_distribution.png")
        plt.close()