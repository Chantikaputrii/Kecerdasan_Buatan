from src.data_mining import load_data
from src.refined_data_cleaning import clean_data
from src.data_exploration import explore_data
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_before_after(before_df, after_df, column, numeric=True):
    plt.figure(figsize=(12,5))
    plt.suptitle(f'Column: {column} - Before and After Cleaning', fontsize=16)

    plt.subplot(1,2,1)
    if numeric:
        sns.histplot(before_df[column].dropna(), kde=True, color='red')
        plt.title('Before Cleaning')
    else:
        before_df[column].value_counts().plot.bar(color='red')
        plt.title('Before Cleaning')

    plt.subplot(1,2,2)
    if numeric:
        sns.histplot(after_df[column].dropna(), kde=True, color='green')
        plt.title('After Cleaning')
    else:
        after_df[column].value_counts().plot.bar(color='green')
        plt.title('After Cleaning')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    plt.pause(0.001)  # Add pause to ensure plot window updates

# Step 1: Mining
df = load_data()

# Step 2: Cleaning
df_clean = clean_data(df)

# Step 3: EDA
explore_data(df_clean)

# Plot before and after cleaning for each column
for col in df_clean.columns:
    numeric = pd.api.types.is_numeric_dtype(df_clean[col])
    plot_before_after(df, df_clean, col, numeric)
