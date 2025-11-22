from src.data_mining import load_data
from src.data_cleaning import clean_data
from src.data_exploration import explore_data

# Step 1: Mining
df = load_data()

# Step 2: Cleaning
df_clean = clean_data(df)

# Step 3: EDA
explore_data(df_clean)
