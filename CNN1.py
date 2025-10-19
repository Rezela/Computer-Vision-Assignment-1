import tensorflow as tf

model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(8, (3, 3), strides=2, padding="same", input_shape=(64, 64, 1)),
        tf.keras.layers.Conv2D(16, (3, 3), strides=2, padding="same"),
        tf.keras.layers.Conv2D(32, (3, 3), strides=1, padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10)
    ]
)

model.summary()