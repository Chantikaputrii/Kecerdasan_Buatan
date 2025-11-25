import pandas as pd

def load_data(path="data/heart_attack_prediction_indonesia.csv"):
    """
    Fungsi untuk memuat dataset dan menampilkan informasi dasar data mining.
    """

    # =============================
    # 1. LOAD DATA
    # =============================
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"ERROR: File '{path}' tidak ditemukan.")
        return None
    except pd.errors.EmptyDataError:
        print(f"ERROR: File '{path}' kosong.")
        return None
    except Exception as e:
        print(f"ERROR: Terjadi kesalahan saat memuat data: {e}")
        return None

    # =============================
    # 2. INFORMASI DATASET
    # =============================

    print("===== SHAPE DATA =====")
    print(f"Jumlah baris : {df.shape[0]}")
    print(f"Jumlah kolom : {df.shape[1]}")

    print("\n===== 5 DATA TERATAS =====")
    print(df.head())

    print("\n===== INFO DATASET =====")
    df.info()

    print("\n===== STATISTIK DESKRIPTIF =====")
    print(df.describe(include="all"))

    return df
