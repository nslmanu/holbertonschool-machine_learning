#!/usr/bin/env python3
"""Module that creates a confusion matrix."""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Create a confusion matrix.

    Args:
        labels (numpy.ndarray): one-hot of shape (m, classes) with the
            correct labels for each data point.
        logits (numpy.ndarray): one-hot of shape (m, classes) with the
            predicted labels.

    Returns:
        numpy.ndarray: confusion matrix of shape (classes, classes) with
            row indices for the correct labels and column indices for the
            predicted labels.
    """
    return np.matmul(labels.T, logits)
