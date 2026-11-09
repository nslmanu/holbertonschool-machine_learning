#!/usr/bin/env python3
"""Calculate the definiteness of a matrix."""
import numpy as np


def definiteness(matrix):
    """Calculates the definiteness of a matrix.

    Args:
        matrix (numpy.ndarray): shape (n, n), the matrix to classify.

    Returns:
        str or None: one of 'Positive definite', 'Positive semi-definite',
        'Negative semi-definite', 'Negative definite', 'Indefinite', or
        None if matrix does not fit any category or is not valid.

    Raises:
        TypeError: if matrix is not a numpy.ndarray.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    if matrix.shape[0] == 0:
        return None
    if not np.array_equal(matrix, matrix.T):
        return None

    w = np.linalg.eigvals(matrix)

    if np.all(w > 0):
        return "Positive definite"
    if np.all(w >= 0):
        return "Positive semi-definite"
    if np.all(w < 0):
        return "Negative definite"
    if np.all(w <= 0):
        return "Negative semi-definite"
    return "Indefinite"
