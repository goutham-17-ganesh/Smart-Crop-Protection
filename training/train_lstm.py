import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def create_lstm(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1)  # Regression output for yield prediction
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_lstm(X_train, y_train, X_val, y_val, input_shape, epochs=50, batch_size=32):
    model = create_lstm(input_shape)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('best_lstm_model.h5', save_best_only=True)
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
# X_train, y_train, X_val, y_val = numpy arrays with time series data
# input_shape = (timesteps, features)
# model, history = train_lstm(X_train, y_train, X_val, y_val, input_shape=(10, 5))
