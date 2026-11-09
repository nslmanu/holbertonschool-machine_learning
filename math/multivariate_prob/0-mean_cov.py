#!/usr/bin/env python3
"""Calculate the mean and covariance of a data set."""
import numpy as np


def mean_cov(X):
    """Calculates the mean and covariance of a data set.

    Args:
        X (numpy.ndarray): shape (n, d), the data set with n data points
            of d dimensions.

    Returns:
        tuple: (mean, cov) where mean has shape (1, d) and cov has
        shape (d, d).

    Raises:
        TypeError: if X is not a 2D numpy.ndarray.
        ValueError: if X contains fewer than 2 data points.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)
    deviation = X - mean
    cov = np.matmul(deviation.T, deviation) / (n - 1)
    return mean, cov
