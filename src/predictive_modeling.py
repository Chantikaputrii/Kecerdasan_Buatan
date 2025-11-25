import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, silhouette_score

from src.feature_engineering import full_feature_engineering_pipeline


def load_and_preprocess_data(path="data/heart_attack_prediction_indonesia.csv"):
    """
    Load raw data and apply full feature engineering pipeline to prepare dataset.
    """
    df = pd.read_csv(path)
    df_processed = full_feature_engineering_pipeline(df, apply_pca=False, select_k=5)
    print("Data loaded and preprocessed. Shape:", df_processed.shape)
    return df_processed


def train_test_split_data(df, target_column="heart_attack", test_size=0.3, random_state=42):
    """
    Split the dataset into train and test sets.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    print(f"Splitting data into train and test sets with test_size={test_size}")
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def evaluate_classification_model(model, X_test, y_test):
    """
    Evaluate classification model on test set and return metrics.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    return acc, prec, rec, y_pred


def plot_confusion_matrix(y_test, y_pred, model_name="Model", output_dir="results"):
    """
    Plot confusion matrix of classification results and save as image file.
    """
    os.makedirs(output_dir, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    filepath = os.path.join(output_dir, f"confusion_matrix_{model_name.replace(' ', '_')}.png")
    plt.savefig(filepath)
    plt.close()
    print(f"Confusion matrix for {model_name} saved to {filepath}")


def cluster_and_evaluate(X, n_clusters=2):
    """
    Perform KMeans clustering and evaluate with silhouette score.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    score = silhouette_score(X, cluster_labels)
    return kmeans, cluster_labels, score


def plot_clusters(X, cluster_labels):
    """
    Plot clusters with scatter plot using the first two features.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=cluster_labels, palette="viridis")
    plt.title("K-Means Clustering")
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.show()


def run_classification_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate KNN, Logistic Regression, and Decision Tree classifiers.
    """
    results = {}

    # KNN
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    acc, prec, rec, y_pred = evaluate_classification_model(knn, X_test, y_test)
    results["KNN"] = {"model": knn, "accuracy": acc, "precision": prec, "recall": rec, "y_pred": y_pred}
    plot_confusion_matrix(y_test, y_pred, model_name="KNN")

    # Logistic Regression
    logreg = LogisticRegression(max_iter=200)
    logreg.fit(X_train, y_train)
    acc, prec, rec, y_pred = evaluate_classification_model(logreg, X_test, y_test)
    results["Logistic Regression"] = {"model": logreg, "accuracy": acc, "precision": prec, "recall": rec, "y_pred": y_pred}
    plot_confusion_matrix(y_test, y_pred, model_name="Logistic Regression")

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    acc, prec, rec, y_pred = evaluate_classification_model(dt, X_test, y_test)
    results["Decision Tree"] = {"model": dt, "accuracy": acc, "precision": prec, "recall": rec, "y_pred": y_pred}
    plot_confusion_matrix(y_test, y_pred, model_name="Decision Tree")

    return results


def run_clustering_model(X):
    """
    Perform KMeans clustering and visualize results.
    """
    kmeans, labels, sil_score = cluster_and_evaluate(X, n_clusters=2)
    print(f"Silhouette Score: {sil_score:.4f}")
    plot_clusters(X, labels)
    return kmeans, labels, sil_score


def main():
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split_data(df, target_column="heart_attack")

    print("=== Running classification models ===")
    classification_results = run_classification_models(X_train, X_test, y_train, y_test)

    print("\n=== Running clustering model ===")
    # Use train+test combined for clustering
    X_all = pd.concat([X_train, X_test])
    run_clustering_model(X_all)


if __name__ == "__main__":
    main()
