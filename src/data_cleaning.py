import pandas as pd

def clean_data(df):
    print("\n===== DATA CLEANING =====")

    # Cek missing values
    print("\nMissing values per kolom:")
    print(df.isnull().sum())

    # Isi missing alcohol_consumption dengan mode
    df['alcohol_consumption'] = df['alcohol_consumption'].fillna(
        df['alcohol_consumption'].mode()[0]
    )

    # Hapus duplikasi
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"\nDuplikasi dihapus: {before - after} baris")

    # Cek tipe data
    print("\nTipe data setelah cleaning:")
    print(df.dtypes)

    # Simpan data bersih
    df.to_csv("data/cleaned_heart_attack_indonesia.csv", index=False)
    print("\nDataset bersih disimpan.")

    return df
