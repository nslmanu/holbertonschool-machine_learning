#!/usr/bin/env python3
"""Module that converts a label vector into a one-hot matrix."""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """Convert a label vector into a one-hot matrix.

    Args:
        labels (numpy.ndarray): the vector of labels to convert.
        classes (int): the number of classes. Inferred if None.

    Returns:
        numpy.ndarray: the one-hot matrix, last dimension is the classes.
    """
    return K.utils.to_categorical(labels, num_classes=classes)
