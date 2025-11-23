import pandas as pd
import numpy as np

def clean_data(df, output_path="data/cleaned_heart_attack_indonesia.csv"):
    """
    Cleans the given dataframe df in place with:
    - Duplicate removal
    - Handling missing values (mean for numeric, mode for categorical)
    - Fixing data types
    - Text normalization
    - Outlier removal (IQR method)
    Saves cleaned df to output_path and returns the cleaned dataframe.
    """
    df = df.copy()  # avoid modifying original

    print("📌 INFO DATA:")
    print(df.info())

    print("\n📌 STATISTIK DATA:")
    desc_num = df.describe()
    desc_cat = df.describe(include="object")
    print(pd.concat([desc_num, desc_cat], axis=1))

    print("\n📌 5 DATA TERATAS:")
    print(df.head())

    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"\n🔧 Duplicate dihapus: {before - after} baris")

    print("\n📌 Jumlah Missing Value awal:")
    print(df.isna().sum())

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())

    print("\n📌 Jumlah Missing Value setelah cleaning:")
    print(df.isna().sum())

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass

    text_cols = df.select_dtypes(include="object").columns
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].str.replace(r"\s+", " ", regex=True)

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    print("\n📌 DATASET FINAL:")
    print(df.head())
    print("\n📌 SHAPE DATASET:", df.shape)

    df.to_csv(output_path, index=False)
    print(f"\n✅ Dataset bersih berhasil disimpan sebagai {output_path}")

    return df
