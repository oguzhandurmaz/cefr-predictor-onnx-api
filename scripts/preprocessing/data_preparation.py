import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def merge_datasets(*data):
    merged_data = pd.concat(data, ignore_index=True, sort=False)
    return merged_data

def label_encode(df, column):
    le = LabelEncoder()
    df["label_encode"] = le.fit_transform(df[column])
    print(le.classes_)
    return df

def get_columns(df, *columns):
    df = df[list(columns)]
    return df

def parse_standart_cefr(value):
    if(value == "A1+"): return "A1"
    if(value == "A2+"): return "A2"
    if(value == "B1+"): return "B1"
    if(value == "B2+"): return "B2"
    return value

def data_preparation():
    df_1 = pd.read_csv("data/raw/cefr_leveled_texts.csv")
    label_encode(df_1, "label")
    df_1 = get_columns(df_1, "text", "label_encode")

    df_2 = pd.read_json("hf://datasets/UniversalCEFR/readme_en/readme_en.json")
    label_encode(df_2, "cefr_level")
    df_2 = get_columns(df_2, "text", "label_encode")

    df_3 = pd.read_json("hf://datasets/UniversalCEFR/cefr_sp_en/cefr-sp.json")
    label_encode(df_3, "cefr_level")
    df_3 = get_columns(df_3, "text", "label_encode")

    df_4 = pd.read_json("hf://datasets/UniversalCEFR/cefr_asag_en/cefr_asag.json")
    label_encode(df_4, "cefr_level")
    df_4 = get_columns(df_4, "text", "label_encode")

    df_5 = pd.read_json("hf://datasets/UniversalCEFR/elg_cefr_en/elg-cefr-en.json")
    df_5["cefr_level"] = df_5["cefr_level"].apply(parse_standart_cefr)
    label_encode(df_5, "cefr_level")
    df_5 = get_columns(df_5, "text", "label_encode")

    merged_data = merge_datasets(df_1, df_2, df_3, df_4, df_5)
    # Save to csv
    merged_data.to_csv("data/processed/merged_data.csv", index=False)
    return merged_data

if __name__ == "__main__":
    data_preparation()
    
    
