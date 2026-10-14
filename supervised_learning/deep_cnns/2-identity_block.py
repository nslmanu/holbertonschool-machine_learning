#!/usr/bin/env python3
"""Build an identity block for a residual network."""

from tensorflow import keras as K


def identity_block(A_prev, filters):
    """Build an identity block.

    Args:
        A_prev: Output tensor from the previous layer.
        filters: Filter counts for the 1x1, 3x3, and 1x1 convolutions.

    Returns:
        The activated output tensor of the identity block.
    """
    F11, F3, F12 = filters

    X = K.layers.Conv2D(
        filters=F11,
        kernel_size=1,
        padding='same',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(
        filters=F3,
        kernel_size=3,
        padding='same',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(
        filters=F12,
        kernel_size=1,
        padding='same',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    X = K.layers.Add()([X, A_prev])
    X = K.layers.Activation('relu')(X)

    return X
