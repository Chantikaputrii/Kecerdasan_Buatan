import matplotlib.pyplot as plt
import seaborn as sns

def explore_data(df):
    print("\n===== DATA EXPLORATION (EDA) =====")
    print(df.describe(include='all'))
    corr = df.corr(numeric_only=True)
    print("\n===== MATRIKS KORELASI =====")
    print(corr)

    plt.figure(figsize=(14, 8))   
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title("Heatmap Korelasi Variabel Numerik", fontsize=14)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.hist(df['age'], bins=30)
    plt.title("Distribusi Umur", fontsize=13)
    plt.xlabel("Umur")
    plt.ylabel("Frekuensi")
    plt.tight_layout()     
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df['blood_pressure_systolic'])
    plt.title("Boxplot Tekanan Darah Sistolik", fontsize=13)
    plt.tight_layout()
    plt.show()

    print("\n===== INSIGHT AWAL =====")
    print("- Rata-rata umur:", df['age'].mean())
    print("- Persentase serangan jantung:", df['heart_attack'].mean() * 100, "%")
    print("- Korelasi tertinggi dengan serangan jantung:")
    print(corr['heart_attack'].sort_values(ascending=False).head())

    return corr