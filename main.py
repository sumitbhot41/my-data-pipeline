import logging
import yaml
import os
import pandas as pd
import joblib
import threading
from concurrent.futures import ThreadPoolExecutor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from joblib import Parallel, delayed
from scripts.data_ingestion import load_csv  # Updated function call
from scripts.preprocessing import preprocess_data
from scripts.eda import perform_eda
from scripts.model_training import train_models
from scripts.visualization import generate_plots
from scripts.hyperparameter import tune_hyperparameters

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Pipeline started.")

# Load configuration
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    data_path = config["data"]["path"]
    model_save_path = config["model"]["save_path"]
    output_folder = "outputs/eda_plots"

    # Ensure output directories exist
    os.makedirs("outputs", exist_ok=True)
    if os.path.exists(output_folder) and not os.path.isdir(output_folder):
        os.remove(output_folder)  # Remove file if it exists
    os.makedirs(output_folder, exist_ok=True)

except (FileNotFoundError, KeyError) as e:
    logging.error(f"Configuration file error: {e}")
    raise

# Step 1: Load data with optimized memory usage
try:
    data = load_csv(data_path)
    logging.info(f"Data loaded successfully with shape {data.shape}.")
except Exception as e:
    logging.error(f"Error loading data: {e}")
    raise

# Step 2: Validate Data
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

# Step 3: Memory-Efficient Preprocessing
def preprocess_data_optimized(data, target_column=None):
    """Efficiently preprocesses data with parallelization and reduced memory usage."""
    if target_column is None:
        print("\nAvailable columns in the dataset:")
        print(list(data.columns))
        target_column = input("\nEnter the target column name: ").strip()

    if target_column not in data.columns:
        raise KeyError(f"Target column '{target_column}' not found.")

    X = data.drop(columns=[target_column])
    y = data[target_column]

    logging.info(f"Selected target column: {target_column}")

    numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Parallelize preprocessing
    X_processed = preprocessor.fit_transform(X)
    X_processed = pd.DataFrame(X_processed, dtype='float32')  # Reduce memory usage

    return X_processed, y.astype('float32')

processed_data, target = preprocess_data_optimized(data)
logging.info("Data preprocessing completed.")

# Step 4: Run Exploratory Data Analysis (Multithreading)
def run_eda():
    try:
        perform_eda(processed_data)
        logging.info("EDA completed.")
    except Exception as e:
        logging.warning(f"EDA skipped due to error: {e}")

eda_thread = threading.Thread(target=run_eda)
eda_thread.start()

# Step 5: Hyperparameter tuning (Multithreading)
def run_hyperparameter_tuning():
    try:
        best_params = tune_hyperparameters(processed_data, target)
        logging.info(f"Best parameters found: {best_params}")
        return best_params
    except Exception as e:
        logging.warning(f"Hyperparameter tuning skipped due to error: {e}")
        return {}

# Run hyperparameter tuning in parallel
with ThreadPoolExecutor() as executor:
    best_params_future = executor.submit(run_hyperparameter_tuning)

# Step 6: Train Model (Multithreading)
def run_model_training():
    try:
        best_params = best_params_future.result()  # Get hyperparameters after tuning completes
        model = train_models(processed_data, target, best_params)
        joblib.dump(model, model_save_path)
        logging.info(f"Model saved at {model_save_path}.")
    except Exception as e:
        logging.error(f"Model training failed: {e}")

# Run model training in parallel
train_thread = threading.Thread(target=run_model_training)
train_thread.start()

# Step 7: Generate Visualizations (Multithreading)
def run_visualization():
    try:
        model = joblib.load(model_save_path)  # Ensure model is loaded
        generate_plots(model, processed_data)
        logging.info("Visualization completed.")
    except Exception as e:
        logging.warning(f"Visualization skipped: {e}")

# Run visualization after training completes
train_thread.join()
eda_thread.join()  # Ensure EDA is complete before visualization
vis_thread = threading.Thread(target=run_visualization)
vis_thread.start()

# Ensure all threads complete
vis_thread.join()

logging.info("Pipeline execution completed.")
