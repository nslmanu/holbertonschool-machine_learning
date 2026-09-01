#!/usr/bin/env python3
"""Module that makes a prediction using a neural network."""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Make a prediction using a neural network.

    Args:
        network (keras.Model): the network model to make the prediction with.
        data (numpy.ndarray): the input data to make the prediction with.
        verbose (bool): whether output should be printed during prediction.

    Returns:
        numpy.ndarray: the prediction for the data.
    """
    return network.predict(data, verbose=verbose)
