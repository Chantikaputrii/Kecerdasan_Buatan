import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def explore_data(df):
    print("\n======================")
    print("===== DATA EXPLORATION (EDA) =====")
    print("======================")

    # ======================================================
    # 1. STATISTIK DESKRIPTIF
    # ======================================================
    print("\n===== 1. STATISTIK DESKRIPTIF =====")

    print("\n📌 Statistik Numerik:")
    print(df.describe())

    print("\n📌 Statistik Kategorikal:")
    print(df.describe(include="object"))

    print("\n📌 Median Setiap Kolom Numerik:")
    print(df.median(numeric_only=True))

    print("\n📌 Mode Setiap Kolom:")
    print(df.mode().iloc[0])


    # ======================================================
    # 2. KORELASI ANTAR VARIABEL
    # ======================================================
    print("\n===== 2. MATRIKS KORELASI =====")

    corr = df.corr(numeric_only=True)
    print(corr)

    plt.figure(figsize=(14, 8))
    sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f")
    plt.title("Heatmap Korelasi Variabel Numerik", fontsize=14)
    plt.tight_layout()
    plt.show()


    # ======================================================
    # 3. VISUALISASI EKSPLORATIF
    # ======================================================
    print("\n===== 3. VISUALISASI EKSPLORATIF =====")

    # ------------------
    # HISTOGRAM UMUR
    # ------------------
    plt.figure(figsize=(8, 5))
    plt.hist(df['age'], bins=30)
    plt.title("Distribusi Umur", fontsize=13)
    plt.xlabel("Umur")
    plt.ylabel("Frekuensi")
    plt.tight_layout()
    plt.show()

    # ------------------
    # BOXPLOT SYSTOLIC BP
    # ------------------
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df['blood_pressure_systolic'])
    plt.title("Boxplot Tekanan Darah Sistolik", fontsize=13)
    plt.tight_layout()
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


    # ======================================================
    # 4. INSIGHT AWAL
    # ======================================================
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


    return corr


# Auto-run jika file dieksekusi langsung
if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_heart_attack_indonesia.csv")
    explore_data(df)
