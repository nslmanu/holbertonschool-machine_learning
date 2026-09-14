#!/usr/bin/env python3
"""Module that calculates the sensitivity for each class."""
import numpy as np


def sensitivity(confusion):
    """Calculate the sensitivity for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): confusion matrix of shape
            (classes, classes) where row indices represent the correct
            labels and column indices represent the predicted labels.

    Returns:
        numpy.ndarray: shape (classes,) with the sensitivity of each class.
    """
    true_positives = np.diagonal(confusion)
    actual_positives = np.sum(confusion, axis=1)
    return true_positives / actual_positives
