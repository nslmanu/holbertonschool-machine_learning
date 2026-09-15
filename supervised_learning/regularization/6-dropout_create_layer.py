#!/usr/bin/env python3
"""Module that creates a TensorFlow layer using dropout."""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """Create a neural network layer using dropout.

    Args:
        prev (tensor): the output of the previous layer.
        n (int): the number of nodes the new layer should contain.
        activation (function): the activation function for the new layer.
        keep_prob (float): the probability that a node will be kept.
        training (bool): whether the model is in training mode.

    Returns:
        tensor: the output of the new layer.
    """
    init = tf.keras.initializers.VarianceScaling(scale=2.0,mode='fan_avg')
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init)
    output = layer(prev)
    dropout = tf.keras.layers.Dropout(rate=1 - keep_prob)
    return dropout(output, training=training)
