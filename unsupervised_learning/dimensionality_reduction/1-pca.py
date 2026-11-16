#!/usr/bin/env python3
"""Perform PCA on a dataset to reduce its dimensionality."""
import numpy as np


def pca(X, ndim):
    """Performs PCA on a dataset.

    Args:
        X (numpy.ndarray): shape (n, d), the dataset with n data points
            of d dimensions.
        ndim (int): the new dimensionality of the transformed X.

    Returns:
        numpy.ndarray of shape (n, ndim): the transformed version of X.
    """
    mean = np.mean(X, axis=0, keepdims=True)
    A = X - mean
    _, _, vh = np.linalg.svd(A)
    W = vh[:ndim].T
    return np.matmul(A, W)
