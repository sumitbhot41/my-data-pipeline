# scripts/preprocessing.py
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def preprocess_data(data, target_column=None):
    """Ask the user to select a target column dynamically if not provided."""
    if target_column is None:
        print("\nAvailable columns in the dataset:")
        print(", ".join(list(data.columns)))  # Display columns properly

        # Ensure the input is a single valid column name
        while True:
            target_column = input("\nEnter the target column name: ").strip()
            if target_column in data.columns:
                break  # Valid input, exit loop
            print(f"Invalid column name. Please select from: {', '.join(data.columns)}")

    logging.info(f"User selected target column: {target_column}")

    X = data.drop(columns=[target_column])  # Features
    y = data[target_column]  # Target variable

    # Define preprocessing for numerical and categorical data
    numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Apply preprocessing
    X_processed = preprocessor.fit_transform(X)
    return X_processed, y
