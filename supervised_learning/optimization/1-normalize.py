#!/usr/bin/env python3
"""Module that normalizes (standardizes) a matrix."""


def normalize(X, m, s):
    """Normalize (standardize) a matrix.

    Args:
        X (numpy.ndarray): matrix of shape (d, nx) to normalize, where d is
            the number of data points and nx is the number of features.
        m (numpy.ndarray): shape (nx,), the mean of all features of X.
        s (numpy.ndarray): shape (nx,), the standard deviation of all
            features of X.

    Returns:
        numpy.ndarray: the normalized X matrix.
    """
    return (X - m) / s
