import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_correlation_heatmap(df, title="Feature Correlation Heatmap"):
    """
    Plots a heatmap showing correlations between features in a DataFrame.
    """
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f', square=True)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_feature_distributions(df, features, title="Feature Distributions"):
    """
    Plots distribution of selected features for exploratory analysis.
    """
    num_features = len(features)
    plt.figure(figsize=(15, 4 * (num_features // 3 + 1)))
    
    for i, feature in enumerate(features, 1):
        plt.subplot((num_features // 3 + 1), 3, i)
        sns.histplot(df[feature], kde=True, color='teal')
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_actual_vs_predicted(y_true, y_pred, model_name="Model"):
    """
    Plots actual vs predicted values for regression outputs.
    """
    plt.figure(figsize=(7, 7))
    sns.scatterplot(x=y_true, y=y_pred, color='purple', alpha=0.6)
    sns.lineplot(x=y_true, y=y_true, color='red', label='Ideal Fit')
    plt.title(f"Actual vs Predicted Yield ({model_name})")
    plt.xlabel("Actual Crop Yield")
    plt.ylabel("Predicted Crop Yield")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_learning_curves(history, model_name="Model"):
    """
    Visualizes training and validation loss over epochs for model training.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(history.history.get('loss', []), label='Training Loss', color='blue')
    plt.plot(history.history.get('val_loss', []), label='Validation Loss', color='orange')
    plt.title(f"Learning Curve: {model_name}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Example usage:
# df = pd.read_csv('engineered_features.csv')
# plot_correlation_heatmap(df)
# plot_feature_distributions(df, ['NDVI', 'EVI', 'temperature', 'rainfall'])
# plot_actual_vs_predicted(y_true, y_pred, model_name="Ensemble Model")
# plot_learning_curves(history, model_name="CNN Model")
