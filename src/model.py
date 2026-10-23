"""Configurable CNN. Augmentation layers live inside the model and only act during training."""
from __future__ import annotations

import config


def build_model(img_size=config.IMG_SIZE, num_classes=config.NUM_CLASSES, filters=config.DEFAULT_FILTERS,
                dense_units=config.DEFAULT_DENSE, dropout=config.DEFAULT_DROPOUT,
                learning_rate=config.DEFAULT_LR, augment=True):
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input((img_size, img_size, 3), name="image")
    x = inputs
    if augment:  # no flips: they could change a sign's meaning (e.g. turn left vs right)
        x = layers.RandomRotation(0.04)(x)
        x = layers.RandomZoom(0.1)(x)
        x = layers.RandomTranslation(0.08, 0.08)(x)
    for f in filters:
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probabilities")(x)
    model = keras.Model(inputs, outputs, name="traffic_sign_cnn")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
