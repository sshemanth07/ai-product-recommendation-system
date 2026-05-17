import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd

def save_plots(predictions, probabilities, y_test, results_dir):
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    sns.set_style("whitegrid")
    
    metrics_df = pd.read_csv(results_dir / "metrics_comparison.csv", index_col=0)
    model_names = list(predictions.keys())
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
    
    # Separate confusion matrix for each model
    for idx, name in enumerate(model_names):
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_test, predictions[name])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    cbar=True, square=True, linewidths=2,
                    xticklabels=['Not Purchased', 'Purchased'],
                    yticklabels=['Not Purchased', 'Purchased'])
        
        plt.title(f'{name} - Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.tight_layout()
        
        # Save individual confusion matrix
        filename = f"{name.lower().replace(' ', '_')}_confusion_matrix.png"
        plt.savefig(plots_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filename}")
    
    # Combined confusion matrices (2x2 grid)
    n_models = len(model_names)
    n_cols = 2
    n_rows = (n_models + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, name in enumerate(model_names):
        cm = confusion_matrix(y_test, predictions[name])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    cbar=False, square=True, linewidths=2,
                    xticklabels=['Not Purchased', 'Purchased'],
                    yticklabels=['Not Purchased', 'Purchased'])
        axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted', fontsize=10)
        axes[idx].set_ylabel('Actual', fontsize=10)
    
    for idx in range(len(model_names), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Confusion Matrices - All Models', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / "all_confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC Curves combined
    plt.figure(figsize=(8, 6))
    for idx, name in enumerate(model_names):
        fpr, tpr, _ = roc_curve(y_test, probabilities[name])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', 
                linewidth=2, color=colors[idx])
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "roc_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Individual ROC curves for each model
    for idx, name in enumerate(model_names):
        plt.figure(figsize=(6, 5))
        fpr, tpr, _ = roc_curve(y_test, probabilities[name])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, linewidth=2, color=colors[idx])
        plt.fill_between(fpr, tpr, alpha=0.3, color=colors[idx])
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'{name} - ROC Curve (AUC = {roc_auc:.3f})', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"{name.lower().replace(' ', '_')}_roc_curve.png"
        plt.savefig(plots_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Performance Metrics Bar Chart
    plt.figure(figsize=(12, 6))
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUROC']
    x = np.arange(len(model_names))
    width = 0.15
    
    for i, metric in enumerate(metric_names):
        values = [metrics_df.loc[name, metric] for name in model_names]
        bars = plt.bar(x + i*width, values, width, label=metric, 
                      edgecolor='black', linewidth=1)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x + width * 2, model_names, rotation=15, ha='right')
    plt.legend(loc='upper right', fontsize=10)
    plt.ylim([0, 1.05])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(plots_dir / "performance_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Metrics Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics_df, annot=True, fmt='.4f', cmap='YlOrRd', 
                cbar_kws={'label': 'Score'}, linewidths=1)
    plt.title('Model Metrics Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / "metrics_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nAll plots saved to {plots_dir}/")
    print(f"Files generated: confusion matrices, ROC curves, performance charts")