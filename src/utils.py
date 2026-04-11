from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def save_plots(predictions, probabilities, y_test, results_dir):
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    metrics_df = pd.read_csv(results_dir / "metrics_comparison.csv", index_col=0)
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "AUROC"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    for i, metric in enumerate(metrics_to_plot):
        sns.barplot(x=metrics_df.index, y=metrics_df[metric], ax=axes[i], palette="viridis")
        axes[i].set_title(metric)
        axes[i].set_ylim(0.5, 1.05)
    plt.tight_layout()
    plt.savefig(plots_dir / "metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, name in zip(axes, predictions.keys()):
        cm = confusion_matrix(y_test, predictions[name])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig(plots_dir / "confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to {plots_dir}/")
