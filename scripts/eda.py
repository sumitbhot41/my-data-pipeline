import os
import logging
import gc
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# Set seaborn style for better plots
sns.set(style="whitegrid")


def perform_eda(data, max_features=10):
    """
    Perform exploratory data analysis (EDA) on the dataset.
    - Generates a pairplot with a limited number of features to avoid memory issues.
    - Creates a heatmap for feature correlation.
    - Saves plots in the `outputs/eda_plots` directory.
    """
    logging.info("Starting EDA...")

    output_folder = "outputs/eda_plots"
    os.makedirs(output_folder, exist_ok=True)

    # Convert sparse data to dense format if necessary
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        logging.warning("Data is in sparse format. Converting to dense DataFrame.")
        df = pd.DataFrame(data.toarray(), dtype='float32')  # Reduce memory usage

    # Drop columns with only one unique value
    df = df.loc[:, df.nunique() > 1]

    # Select top features for pairplot (Limit to `max_features` most varied)
    feature_variances = df.var().sort_values(ascending=False)
    top_features = feature_variances.index[:max_features]

    logging.info(f"Using {len(top_features)} features for pairplot.")

    def generate_pairplot():
        try:
            logging.info("Generating pairplot...")
            sns.pairplot(df[top_features])
            plt.savefig(os.path.join(output_folder, "pairplot.png"), dpi=150)
            plt.close()
            logging.info("Pairplot saved successfully.")
        except Exception as e:
            logging.error(f"Error generating pairplot: {e}")

    def generate_correlation_heatmap():
        try:
            logging.info("Generating correlation heatmap...")
            corr = df.iloc[:, :100].corr()  # Limit to 100 features
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr, cmap="coolwarm", annot=False)
            plt.title("Feature Correlation Heatmap")
            plt.savefig(os.path.join(output_folder, "correlation_heatmap.png"), dpi=150)
            plt.close()
            logging.info("Correlation heatmap saved successfully.")
        except Exception as e:
            logging.error(f"Error generating heatmap: {e}")

    # Run pairplot & heatmap in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(generate_pairplot)
        executor.submit(generate_correlation_heatmap)

    # Free memory after EDA
    del df
    gc.collect()

    logging.info("EDA completed successfully.")
