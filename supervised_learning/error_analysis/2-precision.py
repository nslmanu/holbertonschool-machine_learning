#!/usr/bin/env python3
"""Module that calculates the precision for each class."""
import numpy as np


def precision(confusion):
    """Calculate the precision for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): confusion matrix of shape
            (classes, classes) where row indices represent the correct
            labels and column indices represent the predicted labels.

    Returns:
        numpy.ndarray: shape (classes,) with the precision of each class.
    """
    true_positives = np.diagonal(confusion)
    predicted_positives = np.sum(confusion, axis=0)
    return true_positives / predicted_positives
