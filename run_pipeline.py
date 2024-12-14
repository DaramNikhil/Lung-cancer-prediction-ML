import pandas as pd
import numpy as np
from src.preprocessing import data_preprocessing
from src.model_dev import model_dovelopement


def read_data(data_path):
    return pd.read_csv(data_path)


if __name__ == "__main__":
    data_path = r"D:\FREELANCE_PROJECTS\lung-cancer-prediction\data\survey lung cancer.csv"
    df = read_data(data_path)
    cleaned_df = data_preprocessing(df)
    model_dovelopement(cleaned_df)
