#!/usr/bin/env python3
"""Module that calculates the normalization constants of a matrix."""
import numpy as np


def normalization_constants(X):
    """Calculate the normalization (standardization) constants of a matrix.

    Args:
        X (numpy.ndarray): matrix of shape (m, nx) to normalize, where m is
            the number of data points and nx is the number of features.

    Returns:
        tuple: the mean and standard deviation of each feature, respectively.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std