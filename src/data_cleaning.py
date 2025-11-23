import pandas as pd
import numpy as np

# ============================
# CLEANING FUNCTION (FINAL VERSION)
# ============================
def clean_data(path, output_path="data/cleaned_heart_attack_indonesia.csv"):

    df = pd.read_csv(path)

    # ======================
    # 1. INFO DATA
    # ======================
    print("📌 INFO DATA:")
    print(df.info())

    # ======================
    # 2. STATISTIK DATA (MIRIP CONTOH)
    # ======================
    print("\n📌 STATISTIK DATA:")

    # Statistik numerik + kategorikal digabung seperti contoh
    desc_num = df.describe()
    desc_cat = df.describe(include="object")

    print(pd.concat([desc_num, desc_cat], axis=1))

    # ======================
    # 3. HEAD
    # ======================
    print("\n📌 5 DATA TERATAS:")
    print(df.head())

    # ======================
    # 4. HAPUS DUPLIKAT
    # ======================
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"\n🔧 Duplicate dihapus: {before - after} baris")

    # ======================
    # 5. MISSING VALUE
    # ======================
    print("\n📌 Jumlah Missing Value awal:")
    print(df.isna().sum())

    # Isi missing value sesuai tipe
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())

    print("\n📌 Jumlah Missing Value setelah cleaning:")
    print(df.isna().sum())

    # ======================
    # 6. PERBAIKAN TIPE DATA
    # ======================
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

    # ======================
    # 7. NORMALISASI TEKS
    # ======================
    text_cols = df.select_dtypes(include="object").columns
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].str.replace(r"\s+", " ", regex=True)

    # ======================
    # 8. HAPUS OUTLIER (IQR)
    # ======================
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    # ======================
    # 9. FINAL DATASET
    # ======================
    print("\n📌 DATASET FINAL:")
    print(df.head())

    print("\n📌 SHAPE DATASET:", df.shape)

    df.to_csv(output_path, index=False)
    print(f"\n✅ Dataset bersih berhasil disimpan sebagai {output_path}")

    return df


# Auto-run jika file dieksekusi langsung
if __name__ == "__main__":
    clean_data("data/heart_attack_prediction_indonesia.csv",
               "data/cleaned_heart_attack_indonesia.csv")
