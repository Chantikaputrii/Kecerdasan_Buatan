import pandas as pd

def load_data(path="data/heart_attack_prediction_indonesia.csv"):
    df = pd.read_csv(path)
    print("===== 5 DATA TERATAS =====")
    print(df.head())

    print("\n===== INFO DATASET =====")
    print(df.info())

    print("\n===== STATISTIK DESKRIPTIF =====")
    print(df.describe())

    return df
