#!/usr/bin/env python3
"""Module that creates a batch normalization layer in TensorFlow."""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Create a batch normalization layer for a neural network in TensorFlow.

    Args:
        prev (tensor): the activated output of the previous layer.
        n (int): the number of nodes in the layer to be created.
        activation (function): the activation function to use on the output.

    Returns:
        tensor: the activated output of the layer.
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense = tf.keras.layers.Dense(units=n, kernel_initializer=init)
    z = dense(prev)

    mean, variance = tf.nn.moments(z, axes=0)
    gamma = tf.Variable(tf.ones((1, n)), trainable=True)
    beta = tf.Variable(tf.zeros((1, n)), trainable=True)
    epsilon = 1e-7

    z_norm = tf.nn.batch_normalization(z, mean, variance, beta, gamma,
                                       epsilon)
    return activation(z_norm)
