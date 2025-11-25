import pandas as pd

def load_data(path="data/heart_attack_prediction_indonesia.csv"):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"ERROR: File '{path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"ERROR: File '{path}' is empty.")
        return None
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while loading the data: {e}")
        return None

    print("===== 5 DATA TERATAS =====")
    print(df.head())

    print("\n===== INFO DATASET =====")
    df.info()

    print("\n===== STATISTIK DESKRIPTIF =====")
    print(df.describe())

    return df