import tensorflow as tf
import numpy as np

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(1, (3, 3), strides=1, padding="valid", input_shape=(5, 5, 1), use_bias=False),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, use_bias=False)
])

K = np.array([
    [[[1]], [[0]], [[-1]]],
    [[[1]], [[0]], [[-1]]],
    [[[1]], [[0]], [[-1]]]
], dtype=np.float32).reshape(3, 3, 1, 1)
model.layers[0].set_weights([K])
# ValueError: You called `set_weights(weights)` on layer 'conv2d' with a weight list of length 1, but the layer was expecting 2 weights.
# Must set Conv2D layer with use_bias=False

W = np.array([
    [1], [-2], [1], [-1], [-3], [0], [1], [-1], [1]
], dtype=np.float32).reshape(9, 1)
model.layers[-1].set_weights([W])
# ValueError: You called `set_weights(weights)` on layer 'dense' with a weight list of length 1, but the layer was expecting 2 weights.
# Must set Dense layer with use_bias=False

A = np.array([
    [2, 0, 14, 5, 4],
    [11, 12, 16, 4, 8],
    [12, 6, 12, 3, 12],
    [10, 10, 2, 20, 11],
    [4, 11, 8, 12, 4]
], dtype=np.float32).reshape(1, 5, 5, 1)

output = model(A)
print("Output: ", output.numpy().item())
