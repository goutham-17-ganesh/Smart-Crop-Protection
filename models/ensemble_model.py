import numpy as np

def ensemble_predict(cnn_preds, lstm_preds, dnn_preds, weights=(0.4, 0.3, 0.3)):
    final_pred = (weights[0]*cnn_preds + weights[1]*lstm_preds + weights[2]*dnn_preds)
    return np.mean(final_pred, axis=0)
