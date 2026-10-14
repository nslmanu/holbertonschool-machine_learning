#!/usr/bin/env python3
"""Module that normalizes an output using batch normalization."""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Normalize an unactivated output of a neural network using batch norm.

    Args:
        Z (numpy.ndarray): shape (m, n) to normalize, where m is the number
            of data points and n is the number of features.
        gamma (numpy.ndarray): shape (1, n), the scales for batch norm.
        beta (numpy.ndarray): shape (1, n), the offsets for batch norm.
        epsilon (float): a small number used to avoid division by zero.

    Returns:
        numpy.ndarray: the normalized Z matrix.
    """
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    return gamma * Z_norm + beta
