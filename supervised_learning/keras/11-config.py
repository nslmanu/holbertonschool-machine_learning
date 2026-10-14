#!/usr/bin/env python3
"""Module that saves and loads a Keras model's configuration in JSON."""
import tensorflow.keras as K


def save_config(network, filename):
    """Save a model's configuration in JSON format.

    Args:
        network (keras.Model): the model whose configuration should be saved.
        filename (str): the path of the file to save the configuration to.

    Returns:
        None
    """
    with open(filename, 'w') as f:
        f.write(network.to_json())
    return None


def load_config(filename):
    """Load a model with a specific configuration.

    Args:
        filename (str): the path of the file containing the model's
            configuration in JSON format.

    Returns:
        keras.Model: the loaded model.
    """
    with open(filename, 'r') as f:
        network = K.models.model_from_json(f.read())
    return network
