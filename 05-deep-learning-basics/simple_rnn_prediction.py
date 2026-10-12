# SimpleRNN example for predicting the next number in a sequence.
# This is a basic training example and not a production model.

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Input sequences and target values
X = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
    [5, 6, 7],
    [6, 7, 8],
    [7, 8, 9],
], dtype=float)

y = np.array([4, 5, 6, 7, 8, 9, 10], dtype=float)

# Reshape X to match the RNN input format
X = X.reshape(7, 3, 1)

# Build the model
model = Sequential([
    SimpleRNN(10, activation="relu", input_shape=(3, 1)),
    Dense(1)
])

# Compile the model
model.compile(optimizer="adam", loss="mse")

# Train the model
model.fit(X, y, epochs=10, verbose=0)

# Predict the next number for the new sequence
new_input = np.array([[8, 9, 10]], dtype=float)
new_input = new_input.reshape(1, 3, 1)

prediction = model.predict(new_input, verbose=0)

print("X shape:", X.shape)
print("new_input shape:", new_input.shape)
print("Predicted next number:", prediction[0][0])
