# main.py
from scripts.data_ingestion import load_csv
from scripts.eda import perform_eda
from scripts.preprocessing import preprocess_data
from scripts.model_training import train_models, hyperparameter_tuning
from scripts.visualization import plot_feature_importance

# Load data
data = load_csv("data/data.csv")

# Perform EDA
perform_eda(data)

# Preprocess data
X, y = preprocess_data(data, target_column="target")

# Train models
train_models(X, y)

# Hyperparameter tuning
best_model = hyperparameter_tuning(X, y)

# Visualize results
plot_feature_importance(best_model, feature_names=X.columns.tolist())