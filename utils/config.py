import os

# ==============================
# PROJECT CONFIGURATION SETTINGS
# ==============================

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Ensure directories exist
for directory in [DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ==============================
# DATA SETTINGS
# ==============================
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed')
YIELD_DATA_FILE = os.path.join(DATA_DIR, 'yield_data.csv')
ENGINEERED_FEATURES_FILE = os.path.join(DATA_DIR, 'engineered_features.csv')

TEST_SIZE = 0.2
RANDOM_STATE = 42

# ==============================
# MODEL HYPERPARAMETERS
# ==============================

# CNN settings
CNN_PARAMS = {
    "input_shape": (64, 64, 3),
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001
}

# LSTM settings
LSTM_PARAMS = {
    "input_shape": (10, 5),  # (timesteps, features)
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001
}

# DNN settings
DNN_PARAMS = {
    "input_dim": 20,
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001
}

# ==============================
# ENSEMBLE CONFIGURATION
# ==============================
ENSEMBLE_WEIGHTS = {
    "cnn": 0.4,
    "lstm": 0.3,
    "dnn": 0.3
}

# ==============================
# TRAINING CALLBACKS
# ==============================
EARLY_STOPPING = {
    "monitor": "val_loss",
    "patience": 10,
    "restore_best_weights": True
}

CHECKPOINT_PATHS = {
    "cnn": os.path.join(MODEL_DIR, 'best_cnn_model.h5'),
    "lstm": os.path.join(MODEL_DIR, 'best_lstm_model.h5'),
    "dnn": os.path.join(MODEL_DIR, 'best_dnn_model.h5')
}

# ==============================
# LOGGING CONFIGURATION
# ==============================
LOGGING = {
    "level": "INFO",
    "log_file": os.path.join(RESULTS_DIR, "training.log")
}


# ==============================
# UTILITY FUNCTION EXAMPLE
# ==============================
def print_config_summary():
    print("\n==== CONFIGURATION SUMMARY ====")
    print(f"Data directory         : {DATA_DIR}")
    print(f"Models directory       : {MODEL_DIR}")
    print(f"Results directory      : {RESULTS_DIR}")
    print(f"Test Split Ratio       : {TEST_SIZE}")
    print(f"Random State           : {RANDOM_STATE}")
    print(f"Ensemble Weights       : {ENSEMBLE_WEIGHTS}")
    print("================================\n")


# Example usage:
# from config import print_config_summary
# print_config_summary()
