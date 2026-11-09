#!/usr/bin/env python3
"""Calculate a correlation matrix from a covariance matrix."""
import numpy as np


def correlation(C):
    """Calculates a correlation matrix.

    Args:
        C (numpy.ndarray): shape (d, d), a covariance matrix.

    Returns:
        numpy.ndarray: shape (d, d), the correlation matrix.

    Raises:
        TypeError: if C is not a numpy.ndarray.
        ValueError: if C is not a 2D square matrix.
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    std = np.sqrt(np.diag(C))
    outer = np.outer(std, std)
    return C / outer
