import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def load_data(filepath):
    """Loads the dataset from the specified filepath."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at {filepath}")
    return pd.read_csv(filepath)

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plots a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()

def plot_feature_importance(model, feature_names, title="Feature Importance"):
    """Plots feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        plt.figure(figsize=(6, 4))
        sns.barplot(x=importances, y=feature_names)
        plt.title(title)
        plt.tight_layout()
        return plt.gcf()
    return None
