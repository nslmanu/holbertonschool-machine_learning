#!/usr/bin/env python3
"""Module that sets up RMSProp optimization in TensorFlow."""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """Set up the RMSProp optimization algorithm in TensorFlow.

    Args:
        alpha (float): the learning rate.
        beta2 (float): the RMSProp weight (discounting factor).
        epsilon (float): a small number to avoid division by zero.

    Returns:
        optimizer: the TensorFlow RMSProp optimizer.
    """
    return tf.keras.optimizers.RMSprop(learning_rate=alpha,
                                       rho=beta2,
                                       epsilon=epsilon)
