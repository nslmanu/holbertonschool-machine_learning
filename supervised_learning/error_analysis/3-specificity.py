#!/usr/bin/env python3
"""Module that calculates the specificity for each class."""
import numpy as np


def specificity(confusion):
    """Calculate the specificity for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): confusion matrix of shape
            (classes, classes) where row indices represent the correct
            labels and column indices represent the predicted labels.

    Returns:
        numpy.ndarray: shape (classes,) with the specificity of each class.
    """
    total = np.sum(confusion)
    true_positives = np.diagonal(confusion)
    false_positives = np.sum(confusion, axis=0) - true_positives
    false_negatives = np.sum(confusion, axis=1) - true_positives
    true_negatives = total - true_positives - false_positives \
        - false_negatives
    return true_negatives / (true_negatives + false_positives)
