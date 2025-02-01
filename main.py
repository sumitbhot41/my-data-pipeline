import logging
import scripts.preprocessing
import yaml
import pandas as pd
import joblib
from scripts.data_ingestion import load_csv  # Updated to use correct function
from scripts.preprocessing import preprocess_data
from scripts.eda import perform_eda
from scripts.model_training import train_model
from scripts.visualization import generate_plots
from scripts.hyperparameter import tune_hyperparameters

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Pipeline started.")

try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    data_path = config["data"]["path"]
    model_save_path = config["model"]["save_path"]
except FileNotFoundError:
    logging.error("Configuration file not found. Please check the path.")
    raise
except KeyError as e:
    logging.error(f"Missing key in configuration: {e}")
    raise

# Step 1: Load data
try:
    data = load_csv(data_path)  # Updated function call
    logging.info("Data loaded successfully.")
except FileNotFoundError:
    logging.error("Data file not found. Please check the path in config.yaml.")
    raise
except pd.errors.EmptyDataError:
    logging.error("Data file is empty.")
    raise
except Exception as e:
    logging.error(f"Error loading data: {e}")
    raise

# Step 2: Validate data
def validate_data(data):
    """Check for missing values or inconsistencies."""
    if data.isnull().any().any():
        raise ValueError("Data contains missing values.")
    logging.info("Data validation passed.")

try:
    validate_data(data)
except ValueError as e:
    logging.error(f"Data validation error: {e}")
    raise

# Step 3: Preprocess data
processed_data = preprocess_data(data)
logging.info("Data preprocessing completed.")

# Step 4: Perform EDA
perform_eda(processed_data)
logging.info("Exploratory Data Analysis completed.")

# Step 5: Hyperparameter tuning (optional)
best_params = tune_hyperparameters(processed_data)
logging.info("Hyperparameter tuning completed.")

# Step 6: Train model
model = train_model(processed_data, best_params)
logging.info("Model training completed.")

# Step 7: Save the trained model
joblib.dump(model, model_save_path)
logging.info(f"Model saved at {model_save_path}.")

# Step 8: Generate visualizations
generate_plots(model, processed_data)
logging.info("Visualizations generated successfully.")

logging.info("Pipeline execution completed.")
