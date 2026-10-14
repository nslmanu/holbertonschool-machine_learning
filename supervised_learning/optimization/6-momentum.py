#!/usr/bin/env python3
"""Module that sets up momentum optimization in TensorFlow."""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """Set up the gradient descent with momentum optimization in TensorFlow.

    Args:
        alpha (float): the learning rate.
        beta1 (float): the momentum weight.

    Returns:
        optimizer: the TensorFlow momentum optimizer.
    """
    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
