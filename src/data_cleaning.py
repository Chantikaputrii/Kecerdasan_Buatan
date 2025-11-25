import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# CLEANING FUNCTION (FINAL)
# ===============================
def clean_data(path, output_path="data/cleaned_heart_attack_indonesia.csv"):

    df = pd.read_csv(path)

    # =====================================================
    # 1. IDENTIFIKASI MASALAH DATA
    # =====================================================

    print("\n===== 📌 INFORMASI DATA =====")
    df.info()

    print("\n===== 📌 5 DATA TERATAS =====")
    print(df.head())

    print("\n===== 📌 STATISTIK AWAL =====")
    print(df.describe(include="all"))

    print("\n===== 📌 CEK MISSING VALUE =====")
    print(df.isna().sum())

    print("\n===== 📌 CEK DUPLIKASI =====")
    print("Jumlah duplikasi:", df.duplicated().sum())

    # =====================================================
    # 2. VISUALISASI SEBELUM CLEANING
    # =====================================================

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    plt.figure(figsize=(12, 6))
    df[numeric_cols].boxplot()
    plt.title("Boxplot Sebelum Cleaning (Outlier Check)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    df[numeric_cols].hist(bins=20)
    plt.suptitle("Distribusi Data Sebelum Cleaning")
    plt.tight_layout()
    plt.show()


    # =====================================================
    # 3. LANGKAH-LANGKAH CLEANING
    # =====================================================

    # --- (a) Hapus Duplikasi ---
    before = len(df)
    df = df.drop_duplicates()
    print(f"\n🔧 Duplikasi dihapus: {before - len(df)} baris")

    # --- (b) Tangani Missing Value ---
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())

    print("\n📌 Missing Value Setelah Cleaning:")
    print(df.isna().sum())

    # --- (c) Perbaikan Tipe Data ---
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

    # --- (d) Normalisasi Teks ---
    text_cols = df.select_dtypes(include="object").columns
    for col in text_cols:
        df[col] = df[col].astype(str).str.lower().str.strip()
        df[col] = df[col].str.replace(r"\s+", " ", regex=True)

    # --- (e) Hapus Outlier (IQR) ---
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df = df[(df[col] >= lower) & (df[col] <= upper)]


    # =====================================================
    # 4. VISUALISASI SETELAH CLEANING
    # =====================================================
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    plt.figure(figsize=(12, 6))
    df[numeric_cols].boxplot()
    plt.title("Boxplot Setelah Cleaning")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    df[numeric_cols].hist(bins=20)
    plt.suptitle("Distribusi Data Setelah Cleaning")
    plt.tight_layout()
    plt.show()


    # =====================================================
    # 5. FINAL DATA
    # =====================================================
    print("\n===== 📌 DATASET FINAL =====")
    print(df.head())

    print("\n📌 SHAPE AKHIR:", df.shape)

    df.to_csv(output_path, index=False)
    print(f"\n✅ Dataset bersih disimpan ke: {output_path}")

    return df


# Auto run
if __name__ == "__main__":
    clean_data("data/heart_attack_prediction_indonesia.csv",
               "data/cleaned_heart_attack_indonesia.csv")
