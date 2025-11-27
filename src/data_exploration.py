# =============================
# 1. Import Library
# =============================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import mode

def explore_data(df):
    print("\n======================")
    print("===== DATA EXPLORATION (EDA) =====")
    print("======================")

    print("🎯 Dataset berhasil dimuat (versi sudah cleaning)")
    print(df.head())

    # ==================================
    # 3. Missing Value Check
    # ==================================
    print("\n===== PENGECEKAN MISSING VALUE =====")
    missing_info = df.isnull().sum()
    missing_info = missing_info[missing_info > 0]

    if missing_info.empty:
        print("Tidak ada Missing Value di dalam dataset.")
    else:
        missing_percentage = (df.isnull().sum() / len(df)) * 100
        missing_table = pd.DataFrame({
            'Total Missing': df.isnull().sum(),
            'Percentage': missing_percentage.round(2)
        })
        print(missing_table)

        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        plt.title('Heatmap Missing Values')
        plt.show()

    # =============================
    # 4. Info Dasar
    # =============================
    print("\n===== INFORMASI DATA =====")
    print(df.info())

    print("\n===== DESKRIPSI DATA NUMERIK =====")
    print(df.describe())

    print("\n===== DESKRIPSI DATA SEMUA KOLOM =====")
    print(df.describe(include='all'))

    # =============================
    # 5. Statistik Tambahan
    # =============================
    print("\n===== STATISTIK LENGKAP =====")
    stats_df = pd.DataFrame({
        'mean': df.mean(numeric_only=True),
        'median': df.median(numeric_only=True),
        'mode': df.mode().iloc[0],
        'variance': df.var(numeric_only=True),
        'std_dev': df.std(numeric_only=True),
        'skewness': df.skew(numeric_only=True),
        'kurtosis': df.kurtosis(numeric_only=True)
    })
    print(stats_df)

    # =============================
    # 6. Histogram Distribusi
    # =============================
    plt.figure(figsize=(8, 5))
    plt.hist(df['age'], bins=30)
    plt.title("Distribusi Umur", fontsize=13)
    plt.xlabel("Umur")
    plt.ylabel("Frekuensi")
    plt.tight_layout()
    plt.show()

    # =============================
    # 7. Boxplot Outlier Check
    # =============================
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df['blood_pressure_systolic'])
    plt.title("Boxplot Tekanan Darah Sistolik", fontsize=13)
    plt.tight_layout()
    plt.show()

    # =============================
    # 8. Heatmap Korelasi
    # =============================
    plt.figure(figsize=(14, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f")
    plt.title("Heatmap Korelasi Variabel Numerik", fontsize=14)
    plt.tight_layout()
    plt.show()

    # =============================
    # 9. Pairplot
    # =============================
    try:
        sns.pairplot(df.select_dtypes(include=['int64','float64']))
        plt.show()
    except:
        print("⚠ Pairplot dilewati (dataset terlalu besar / RAM penuh)")

    # =============================
    # 10. Analisis Kategorikal
    # =============================
    cat_cols = df.select_dtypes(include=['object']).columns

    for col in cat_cols:
        print(f"\nFrekuensi: {col}")
        print(df[col].value_counts())

        plt.figure(figsize=(7,4))
        sns.countplot(x=df[col])
        plt.title(f'Countplot: {col}')
        plt.xticks(rotation=45)
        plt.show()

    # =============================
    # 11. Perbandingan Smoker berdasarkan Gender
    # =============================
    plt.figure(figsize=(8,5))
    sns.countplot(x='sex', hue='smoker', data=df)
    plt.title("Perbandingan Jumlah Perokok Berdasarkan Gender")
    plt.xlabel("Gender")
    plt.ylabel("Jumlah")
    plt.legend(title="Smoker", labels=["Tidak Merokok","Merokok"])
    plt.show()

    # ------------------
    # SCATTERPLOT AGE vs CHOLESTEROL
    # ------------------
    if 'cholesterol' in df.columns:
        plt.figure(figsize=(7, 5))
        sns.scatterplot(x=df['age'], y=df['cholesterol'])
        plt.title("Scatter Plot: Umur vs Kolesterol", fontsize=13)
        plt.xlabel("Umur")
        plt.ylabel("Kolesterol")
        plt.tight_layout()
        plt.show()

    print("\n===== 4. INSIGHT AWAL =====")

    print("- Rata-rata umur penduduk:", round(df['age'].mean(), 2))
    print("- Median umur:", df['age'].median())
    print("- Persentase serangan jantung:", round(df['heart_attack'].mean() * 100, 2), "%")

    print("\n- Korelasi tertinggi terhadap serangan jantung:")
    print(corr['heart_attack'].sort_values(ascending=False).head())

    if 'cholesterol' in df.columns:
        print("\n- Pola Umur vs Kolesterol:")
        print("  Semakin tua usia, cenderung kolesterol meningkat (cek scatter plot).")

    print("\n- Outlier terlihat jelas pada boxplot tekanan darah sistolik.")

    print("\n🎉 EDA SELESAI")

    return corr


# Auto-run jika file dieksekusi langsung
if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_heart_attack_indonesia.csv")
    explore_data(df)
