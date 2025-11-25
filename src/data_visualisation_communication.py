import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.predictive_modeling import load_and_preprocess_data, train_test_split_data, run_classification_models, run_clustering_model


def plot_confusion_matrix_static(y_test, y_pred, model_name="Model", output_dir="results"):
    """
    Plot and save confusion matrix as static image.
    """
    os.makedirs(output_dir, exist_ok=True)
    cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'], dropna=False)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    filepath = os.path.join(output_dir, f"confusion_matrix_{model_name.replace(' ', '_')}.png")
    plt.savefig(filepath)
    plt.close()
    print(f"Confusion matrix for {model_name} saved to {filepath}")


def create_confusion_matrix_plotly(y_test, y_pred, model_name="Model"):
    """
    Create an interactive confusion matrix heatmap using plotly.
    """
    cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'], dropna=False)
    fig = go.Figure(data=go.Heatmap(
        z=cm.values,
        x=cm.columns.astype(str),
        y=cm.index.astype(str),
        colorscale='Blues',
        hoverongaps=False,
        text=cm.values,
        texttemplate="%{text}",
        showscale=True
    ))
    fig.update_layout(
        title_text=f"Confusion Matrix - {model_name}",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        yaxis_autorange='reversed'
    )
    return fig


def plot_model_metrics(results):
    """
    Plot bar charts for accuracy, precision, and recall for each classification model.
    """
    models = list(results.keys())
    accuracy = [results[m]['accuracy'] for m in models]
    precision = [results[m]['precision'] for m in models]
    recall = [results[m]['recall'] for m in models]

    fig = make_subplots(rows=1, cols=3, subplot_titles=["Accuracy", "Precision", "Recall"])

    fig.add_trace(go.Bar(x=models, y=accuracy, name="Accuracy"), row=1, col=1)
    fig.add_trace(go.Bar(x=models, y=precision, name="Precision"), row=1, col=2)
    fig.add_trace(go.Bar(x=models, y=recall, name="Recall"), row=1, col=3)

    fig.update_layout(title_text="Classification Model Metrics Comparison", showlegend=False)
    return fig


def plot_clusters_plotly(X, cluster_labels):
    """
    Plot interactive scatter plot of clusters using the first two features.
    """
    df_plot = X.copy()
    df_plot['Cluster'] = cluster_labels.astype(str)
    fig = px.scatter(df_plot, x=X.columns[0], y=X.columns[1], color='Cluster', 
                     title="K-Means Clustering Interactive Plot", 
                     labels={X.columns[0]: X.columns[0], X.columns[1]: X.columns[1]})
    return fig


def interpret_results(results):
    """
    Interpret the classification results and provide textual summary.
    """
    interpretation = []
    interpretation.append("Classification Models Performance Interpretation:")
    for model_name, metrics in results.items():
        acc = metrics['accuracy']
        prec = metrics['precision']
        rec = metrics['recall']
        interpretation.append(
            f"- {model_name}: Accuracy={acc:.2f}, Precision={prec:.2f}, Recall={rec:.2f}. "
            f"This suggests the model's ability to correctly classify heart attack cases with these metrics."
        )
    return "\n".join(interpretation)


def insights_and_recommendations(results, silhouette_score):
    """
    Provide insights and recommendations based on model performance and clustering analysis.
    """
    insights = [
        "Insights and Recommendations:",
        f"- The silhouette score for clustering is {silhouette_score:.3f}, indicating the cluster cohesion and separation.",
        "- Among the classification models tested, select the model with the best balance of precision and recall for deployment.",
        "- Models with high recall are preferred if minimizing false negatives (missed heart attack cases) is critical.",
        "- Consider investigating feature importance in the best performing classification model for actionable clinical insights.",
        "- Use cluster analysis to identify patient subgroups for tailored interventions.",
        "- Further data collection and feature engineering may improve model performance.",
    ]
    return "\n".join(insights)


def main():
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split_data(df, target_column="heart_attack")

    # Train classification models
    classification_results = run_classification_models(X_train, X_test, y_train, y_test)

    # Save static confusion matrices for all models
    for model_name, res in classification_results.items():
        plot_confusion_matrix_static(y_test, res['y_pred'], model_name=model_name)

    # Prepare interactive plots
    cm_figs = {}
    for model_name, res in classification_results.items():
        cm_figs[model_name] = create_confusion_matrix_plotly(y_test, res['y_pred'], model_name)

    metrics_fig = plot_model_metrics(classification_results)

    # Run clustering on full dataset
    X_full = pd.concat([X_train, X_test])
    kmeans, cluster_labels, sil_score = run_clustering_model(X_full)
    cluster_fig = plot_clusters_plotly(X_full, cluster_labels)

    # Display interpretation and insights
    interpretation_text = interpret_results(classification_results)
    recommendation_text = insights_and_recommendations(classification_results, sil_score)

    # Show plots (static matplotlib plots already saved)
    print("\n--- Interpretation ---")
    print(interpretation_text)

    print("\n--- Insights and Recommendations ---")
    print(recommendation_text)

    # Interactive plots can be shown in notebook or saved to html
    print("\nSaving interactive plots to html files in 'results' directory...")
    os.makedirs("results", exist_ok=True)
    for model_name, fig in cm_figs.items():
        html_path = os.path.join("results", f"interactive_confusion_matrix_{model_name.replace(' ', '_')}.html")
        fig.write_html(html_path)
        print(f"Saved interactive confusion matrix for {model_name} to {html_path}")

    metrics_html_path = os.path.join("results", "interactive_model_metrics.html")
    metrics_fig.write_html(metrics_html_path)
    print(f"Saved interactive model metrics comparison to {metrics_html_path}")

    cluster_html_path = os.path.join("results", "interactive_cluster_plot.html")
    cluster_fig.write_html(cluster_html_path)
    print(f"Saved interactive cluster plot to {cluster_html_path}")


if __name__ == "__main__":
    main()
