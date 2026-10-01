import numpy as np
from sklearn.model_selection import train_test_split
from processing import process_data, data
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Dropout, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential

X, Y = process_data()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
Y_train_encoded = to_categorical(Y_train)
Y_test_encoded = to_categorical(Y_test)

checkpoint = ModelCheckpoint(
    "best_model.keras",
    monitor='val_loss',
    save_best_only=True,
    mode='min',
)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(30, 126)))
model.add(LayerNormalization())
model.add(Dropout(0.5))

model.add(LSTM(128, return_sequences=False))
model.add(LayerNormalization())
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu', kernel_regularizer='l2'))

num_classes = len(data)
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_train, Y_train_encoded, 
          validation_data=(X_test, Y_test_encoded), 
          epochs=100, 
          batch_size=32,
          callbacks=[early_stopping, checkpoint]
)