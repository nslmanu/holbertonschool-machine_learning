#!/usr/bin/env python3
"""Build the ResNet-50 architecture."""

from tensorflow import keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Build ResNet-50 for inputs of shape (224, 224, 3).

    Returns:
        A Keras model with a 1000-class softmax output.
    """
    inputs = K.Input(shape=(224, 224, 3))

    X = K.layers.Conv2D(
        filters=64,
        kernel_size=7,
        strides=2,
        padding='same',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(inputs)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding='same'
    )(X)

    # Stage 1: 3 residual blocks.
    X = projection_block(X, [64, 64, 256], s=1)
    for _ in range(2):
        X = identity_block(X, [64, 64, 256])

    # Stage 2: 4 residual blocks.
    X = projection_block(X, [128, 128, 512], s=2)
    for _ in range(3):
        X = identity_block(X, [128, 128, 512])

    # Stage 3: 6 residual blocks.
    X = projection_block(X, [256, 256, 1024], s=2)
    for _ in range(5):
        X = identity_block(X, [256, 256, 1024])

    # Stage 4: 3 residual blocks.
    X = projection_block(X, [512, 512, 2048], s=2)
    for _ in range(2):
        X = identity_block(X, [512, 512, 2048])

    X = K.layers.AveragePooling2D(pool_size=7)(X)
    outputs = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(X)

    return K.Model(inputs=inputs, outputs=outputs)
