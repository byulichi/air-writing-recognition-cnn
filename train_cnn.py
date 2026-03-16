import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

print("1. Loading EMNIST Dataset from CSV...")
# UPDATE THESE NAMES IF YOUR FILES ARE NAMED DIFFERENTLY
train_data = pd.read_csv('emnist-letters-train.csv', header=None)
test_data = pd.read_csv('emnist-letters-test.csv', header=None)

print("2. Preprocessing Data (Fixing the EMNIST Trap)...")
# EMNIST 'letters' labels are 1-26. We subtract 1 to make them 0-25 for Python.
y_train = train_data.iloc[:, 0].values - 1
y_test = test_data.iloc[:, 0].values - 1

# Convert labels to categorical (One-Hot Encoding)
y_train = to_categorical(y_train, num_classes=26)
y_test = to_categorical(y_test, num_classes=26)

# Extract pixel data and reshape to 28x28
x_train = train_data.iloc[:, 1:].values.reshape(-1, 28, 28)
x_test = test_data.iloc[:, 1:].values.reshape(-1, 28, 28)

# FIXING THE EMNIST TRAP: Transpose the images so they are upright and readable
x_train = np.transpose(x_train, axes=[0, 2, 1])
x_test = np.transpose(x_test, axes=[0, 2, 1])

# Add the color channel dimension (grayscale = 1) and normalize pixels to 0-1
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

print("3. Building the CNN Architecture...")
model = Sequential([
    # First Convolutional Block (Looking for edges and curves)
    Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2,2),
    
    # Second Convolutional Block (Looking for complex shapes like circles in b, d, p, q)
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    # Flatten and Decision Making
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # Prevents the model from memorizing the data
    Dense(26, activation='softmax') # 26 output neurons for 26 letters
])

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

print("4. Training 'The Brain'...")

# The FYP Flex: Early Stopping
# Patience=5 means "if you don't improve for 5 epochs, stop early"
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# We set a "convincing amount" of 50 epochs, but the AI will smartly stop itself when it peaks.
history = model.fit(x_train, y_train, 
                    epochs=50, 
                    validation_data=(x_test, y_test),
                    batch_size=128,
                    callbacks=[early_stop])

print("5. Saving the Model...")
model.save('dyslexia_air_writing_model.h5')
print("Training Complete! The brain is saved as 'dyslexia_air_writing_model.h5'")