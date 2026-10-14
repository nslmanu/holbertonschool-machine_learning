#!/usr/bin/env python3
"""Module that shuffles the data points in two matrices the same way."""
import numpy as np


def shuffle_data(X, Y):
    """Shuffle the data points in two matrices the same way.

    Args:
        X (numpy.ndarray): first matrix of shape (m, nx) to shuffle.
        Y (numpy.ndarray): second matrix of shape (m, ny) to shuffle.

    Returns:
        tuple: the shuffled X and Y matrices.
    """
    permutation = np.random.permutation(X.shape[0])
    return X[permutation], Y[permutation]
