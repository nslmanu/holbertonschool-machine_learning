#!/usr/bin/env python3
"""Module that tests a neural network."""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Test a neural network.

    Args:
        network (keras.Model): the network model to test.
        data (numpy.ndarray): the input data to test the model with.
        labels (numpy.ndarray): the correct one-hot labels of data.
        verbose (bool): whether output should be printed during testing.

    Returns:
        list: the loss and accuracy of the model with the testing data.
    """
    return network.evaluate(data, labels, verbose=verbose)
