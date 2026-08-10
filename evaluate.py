import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import config

def evaluate_model(model_name, y_true, y_pred):
    """
    Calculates basic classification metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    # Use macro average for multiclass
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    return {
        'Model': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    }

def plot_confusion_matrix(model_name, y_true, y_pred):
    """
    Generates and saves a confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    
    class_names = [config.CLASSES.get(i, f"Class {i}") for i in range(config.NUM_CLASSES)]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save the plot
    filename = f"{model_name.replace(' ', '_')}_confusion_matrix.png"
    filepath = os.path.join(config.PLOT_SAVE_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved confusion matrix to {filepath}")
    
def print_summary_table(results_list):
    """
    Prints a formatted summary table of all models' performance.
    """
    print("\n" + "="*60)
    print(f"{'Model':<20} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8}")
    print("-" * 60)
    
    # Sort by F1-Score descending
    sorted_results = sorted(results_list, key=lambda x: x['F1-Score'], reverse=True)
    
    for res in sorted_results:
        print(f"{res['Model']:<20} | {res['Accuracy']:.4f}   | {res['Precision']:.4f}    | {res['Recall']:.4f}   | {res['F1-Score']:.4f}")
    print("="*60 + "\n")

def plot_feature_importance(model_name, model):
    """
    Plots feature importances for tree-based models.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        features = config.FEATURE_COLS
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Feature Importances - {model_name}')
        plt.bar(range(len(features)), importances[indices], align="center")
        plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45, ha='right')
        plt.xlim([-1, len(features)])
        plt.tight_layout()
        
        filename = f"{model_name.replace(' ', '_')}_feature_importance.png"
        filepath = os.path.join(config.PLOT_SAVE_DIR, filename)
        plt.savefig(filepath)
        plt.close()
        print(f"Saved feature importance plot to {filepath}")

