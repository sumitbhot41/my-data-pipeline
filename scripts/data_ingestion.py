import pandas as pd
def load_csv(file_path):
    return pd.read_csv(file_path)

def load_excel(file_path, sheet_name=0):
    return pd.read_excel(file_path, sheet_name=sheet_name)

def load_xml(file_path):
    return pd.read_xml(file_path)


csv_data = load_csv("data.csv")
excel_data = load_excel("data.xlsx")
xml_data = load_xml("data.xml")