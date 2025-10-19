import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def create_dnn(input_dim):
    model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)  # Regression output for yield
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_dnn(X_train, y_train, X_val, y_val, input_dim, epochs=50, batch_size=32):
    model = create_dnn(input_dim)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('best_dnn_model.h5', save_best_only=True)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    return model, history

# Example usage:
# Assuming X_train, y_train, X_val, y_val are numpy arrays
# input_dim = X_train.shape[1]
# model, history = train_dnn(X_train, y_train, X_val, y_val, input_dim)
