#!/usr/bin/env python3
"""Module that creates a TensorFlow layer with L2 regularization."""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Create a neural network layer in TensorFlow with L2 regularization.

    Args:
        prev (tensor): the output of the previous layer.
        n (int): the number of nodes the new layer should contain.
        activation (function): the activation function for the layer.
        lambtha (float): the L2 regularization parameter.

    Returns:
        tensor: the output of the new layer.
    """
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init,
        kernel_regularizer=tf.keras.regularizers.l2(lambtha))
    return layer(prev)
