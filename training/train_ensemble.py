import numpy as np

def ensemble_predict(cnn_model, lstm_model, dnn_model, X_cnn, X_lstm, X_dnn, weights=[0.4, 0.3, 0.3]):
    """
    Ensemble predictions from CNN, LSTM, and DNN models.

    Parameters:
    - cnn_model, lstm_model, dnn_model: trained models
    - X_cnn, X_lstm, X_dnn: inputs matching each model's expected input format
    - weights: list of weights for each model's prediction

    Returns:
    - combined_predictions: weighted average predictions
    """
    preds_cnn = cnn_model.predict(X_cnn).flatten()
    preds_lstm = lstm_model.predict(X_lstm).flatten()
    preds_dnn = dnn_model.predict(X_dnn).flatten()

    combined_predictions = (
        weights[0] * preds_cnn +
        weights[1] * preds_lstm +
        weights[2] * preds_dnn
    )

    return combined_predictions

# Example usage after loading your trained models and dataset splits:
# combined_preds = ensemble_predict(cnn_model, lstm_model, dnn_model, X_cnn_test, X_lstm_test, X_dnn_test)
# You can then evaluate combined_preds against true yields using metrics like MAE, RMSE, R2, etc.

# Additionally, to optimize weights, consider grid search or meta-model training for blending.
