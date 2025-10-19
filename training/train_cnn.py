import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def create_cnn(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1)  # Regression output for yield prediction
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_cnn(X_train, y_train, X_val, y_val, input_shape, epochs=50, batch_size=32):
    model = create_cnn(input_shape)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('best_cnn_model.h5', save_best_only=True)
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
# Assuming X_train, y_train, X_val, y_val are numpy arrays or tensors
# and input_shape matches the shape of your image or satellite data

# model, history = train_cnn(X_train, y_train, X_val, y_val, input_shape=(64, 64, 3))
