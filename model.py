import tensorflow as tf

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Input(shape=(64,64,3)),

        tf.keras.layers.Conv2D(32,(2,2),activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64,(2,2),activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(128,(2,2),activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(256,activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(64,activation='relu',kernel_regularizer=tf.keras.regularizers.l1_l2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(4,activation='softmax')

    ]
)