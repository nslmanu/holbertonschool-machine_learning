#!/usr/bin/env python3
"""Module that calculates the cost of a network with L2 regularization."""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """Calculate the cost of a neural network with L2 regularization.

    Args:
        cost (tf.Tensor): the cost of the network without L2 regularization.
        model (tf.keras.Model): a Keras model with layers that include L2
            regularization.

    Returns:
        tf.Tensor: the total cost for each layer of the network, accounting
            for L2 regularization.
    """
    return cost + tf.stack(model.losses)
