#!/usr/bin/env python3
"""Module that creates a learning rate decay operation in TensorFlow."""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """Create a learning rate decay operation using inverse time decay.

    The decay occurs in a stepwise fashion.

    Args:
        alpha (float): the original learning rate.
        decay_rate (float): the weight used to determine the decay rate.
        decay_step (int): the number of passes before alpha is decayed
            further.

    Returns:
        the learning rate decay operation.
    """
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True)
