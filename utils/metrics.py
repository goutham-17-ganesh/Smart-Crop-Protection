import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_regression(y_true, y_pred):
    """
    Evaluate regression model performance using standard metrics.

    Parameters:
    - y_true: array-like of actual target values
    - y_pred: array-like of predicted target values

    Returns:
    - metrics: dictionary containing MAE, MSE, RMSE, and R²
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    metrics = {
        "Mean Absolute Error (MAE)": mae,
        "Mean Squared Error (MSE)": mse,
        "Root Mean Squared Error (RMSE)": rmse,
        "R² Score": r2
    }

    return metrics


def print_metrics(metrics_dict, model_name="Model"):
    """
    Print formatted metrics for easy comparison.
    """
    print(f"\nEvaluation Metrics for {model_name}:")
    print("-" * 45)
    for key, value in metrics_dict.items():
        print(f"{key:35}: {value:.4f}")
    print("-" * 45)


# Example usage:
# from metrics import evaluate_regression, print_metrics
# y_true = [2.5, 3.0, 4.2, 5.1]
# y_pred = [2.6, 2.9, 4.1, 5.0]
# results = evaluate_regression(y_true, y_pred)
# print_metrics(results, model_name="Ensemble Model")
