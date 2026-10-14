#!/usr/bin/env python3
"""Module that saves and loads a Keras model's weights."""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """Save a model's weights."""
    try:
        network.save_weights(filename, save_format=save_format)
    except TypeError:
        network.save_weights(filename)
    return None


def load_weights(network, filename):
    """Load a model's weights.

    Args:
        network (keras.Model): the model to which the weights should be loaded.
        filename (str): the path of the file to load the weights from.

    Returns:
        None
    """
    network.load_weights(filename)
    return None
