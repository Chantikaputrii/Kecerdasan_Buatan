import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2



# ======================================================
# 1. CREATE NEW FEATURES
# ======================================================
def create_new_features(df):
    """
    Membuat fitur baru:
    - age_group: mengelompokkan umur → mempermudah model mengenali usia risiko
    - bp_diff: selisih tekanan darah → indikator kesehatan jantung
    - cholesterol_level: kategori kolesterol → penyederhanaan fitur
    """

    df = df.copy()

    # --- Age group ---
    bins = [0, 30, 50, 120]
    labels = ['young', 'middle_aged', 'senior']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

    # --- Blood pressure difference ---
    if 'blood_pressure_systolic' in df.columns and 'blood_pressure_diastolic' in df.columns:
        df['bp_diff'] = df['blood_pressure_systolic'] - df['blood_pressure_diastolic']

    # --- Cholesterol level category ---
    if 'cholesterol' in df.columns:
        df['cholesterol_level'] = pd.cut(
            df['cholesterol'], 
            bins=[0, 200, np.inf], 
            labels=['normal', 'high'], 
            right=False
        )

    return df



# ======================================================
# 2. TRANSFORM FEATURES (SCALING & ENCODING & PCA)
# ======================================================
def transform_features(df, numeric_features=None, categorical_features=None, apply_pca=False):
    """
    Scaling, encoding, dan PCA.
    - StandardScaler untuk numeric
    - OneHotEncoder untuk kategori
    - PCA optional untuk reduksi dimensi
    """

    df = df.copy()

    # --- Scaling ---
    if numeric_features:
        scaler = StandardScaler()
        df[numeric_features] = scaler.fit_transform(df[numeric_features])

    # --- Encoding ---
    if categorical_features:
        encoder = OneHotEncoder(sparse=False, drop='first')
        encoded = encoder.fit_transform(df[categorical_features])
        encoded_df = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out(categorical_features),
            index=df.index
        )
        df = pd.concat([df.drop(columns=categorical_features), encoded_df], axis=1)

    # --- PCA (optional) ---
    if apply_pca:
        pca = PCA(n_components=0.9)  # menjaga 90% variansi
        pca_array = pca.fit_transform(df)
        df = pd.DataFrame(pca_array, index=df.index)
        print(f"\n📌 PCA mengubah data menjadi {df.shape[1]} komponen utama")

    return df



# ======================================================
# 3. FEATURE SELECTION (SELECTKBEST)
# ======================================================
def select_features(df, target_column, k=5):
    """
    Memilih fitur terbaik menggunakan SelectKBest (chi-square).
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    selector = SelectKBest(score_func=chi2, k=k)
    selector.fit(X, y)

    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask]

    print("\n===== 📌 FITUR TERPILIH (SELECTKBEST) =====")
    print(selected_features.tolist())

    return df[selected_features], y



# ======================================================
# 4. FULL PIPELINE
# ======================================================
def full_feature_engineering_pipeline(df, apply_pca=False, select_k=5):
    """
    Pipeline lengkap:
    - Membuat fitur baru
    - Transformasi (scaling + encoding + optional PCA)
    - Pemilihan fitur terbaik
    """

    # --- Step 1: new features ---
    df = create_new_features(df)

    # --- Step 2: transform ---
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'heart_attack' in numerical_cols:
        numerical_cols.remove('heart_attack')

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    df = transform_features(
        df,
        numeric_features=numerical_cols,
        categorical_features=categorical_cols,
        apply_pca=apply_pca
    )

    # --- Step 3: Feature selection ---
    if select_k > 0 and not apply_pca:
        X_selected, y = select_features(df, target_column="heart_attack", k=select_k)
        return pd.concat([X_selected, y], axis=1)

    return df
