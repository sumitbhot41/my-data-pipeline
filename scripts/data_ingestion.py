import pandas as pd
import os
import logging
from scripts.data_ingestion import load_data
from scripts.data_ingestion import load_csv


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_csv(file_path):
    """Load data from a CSV file."""
    try:
        logging.info(f"Loading CSV file from: {file_path}")
        return pd.read_csv(file_path)
    except FileNotFoundError:
        logging.error(f"CSV file not found at: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading CSV file: {e}")
        raise

def load_excel(file_path, sheet_name=0):
    """Load data from an Excel file."""
    try:
        logging.info(f"Loading Excel file from: {file_path}")
        return pd.read_excel(file_path, sheet_name=sheet_name)
    except FileNotFoundError:
        logging.error(f"Excel file not found at: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading Excel file: {e}")
        raise

def load_xml(file_path):
    """Load data from an XML file."""
    try:
        logging.info(f"Loading XML file from: {file_path}")
        return pd.read_xml(file_path)
    except FileNotFoundError:
        logging.error(f"XML file not found at: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading XML file: {e}")
        raise

# Example usage
if __name__ == "__main__":
    # Define file paths (relative or absolute)
    data_dir = "data"  # Folder where data files are stored
    csv_file = os.path.join(data_dir, "data.csv")
    excel_file = os.path.join(data_dir, "data.xlsx")
    xml_file = os.path.join(data_dir, "data.xml")

    try:
        # Load data
        csv_data = load_csv(csv_file)
        excel_data = load_excel(excel_file)
        xml_data = load_xml(xml_file)

        # Print sample data (for debugging)
        print("CSV Data Sample:")
        print(csv_data.head())

        print("Excel Data Sample:")
        print(excel_data.head())

        print("XML Data Sample:")
        print(xml_data.head())

    except Exception as e:
        logging.error(f"Error in data ingestion: {e}")