import numpy as np
import pandas as pd
import sklearn
import scipy
import joblib
import yaml
import matplotlib
import seaborn as sns
import xgboost
import tqdm

# Creating a dictionary of package versions
packages = {
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit-learn": sklearn.__version__,
    "scipy": scipy.__version__,
    "joblib": joblib.__version__,
    "pyyaml": yaml.__version__,
    "matplotlib": matplotlib.__version__,
    "seaborn": sns.__version__,
    "xgboost": xgboost.__version__,
    "tqdm": tqdm.__version__
}

# Printing versions
for pkg, version in packages.items():
    print(f"!pip install {pkg}=={version}")
