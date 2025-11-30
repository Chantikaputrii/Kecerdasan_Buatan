# clustering script
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Non-interactive backend
plt.switch_backend('Agg')

df = pd.read_csv("data/cleaned_heart_attack_indonesia.csv")

N_SAMPLE = 5000
if len(df) > N_SAMPLE:
    df_sample = df.sample(N_SAMPLE, random_state=42).reset_index(drop=True)
else:
    df_sample = df.copy()

X = df_sample.drop(columns=["heart_attack"])
X_encoded = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5, max_iter=200)
df_sample["cluster"] = kmeans.fit_predict(X_scaled)

numeric_cols_all = df_sample.select_dtypes(include=["int64", "float64"]).columns.tolist()
cluster_summary = df_sample.groupby("cluster")[numeric_cols_all].mean().round(2)
cluster_counts = df_sample["cluster"].value_counts().sort_index()

print("Cluster counts:\n", cluster_counts)
print("Cluster summary:\n", cluster_summary)

# Visualisasi clustering dengan PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 5))

# Plot 1: Scatter plot dengan PCA
plt.subplot(1, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df_sample["cluster"], cmap="viridis", s=50, alpha=0.6)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.title('K-Means Clustering (PCA Projection)')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)

# Plot 2: Bar chart cluster counts
plt.subplot(1, 2, 2)
cluster_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Cluster Counts')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('clustering_result.png', dpi=100, bbox_inches='tight')
print("\nGrafik disimpan sebagai 'clustering_result.png'")
plt.close()
