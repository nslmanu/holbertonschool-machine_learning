#!/usr/bin/env python3
"""Perform principal component analysis with NumPy."""

import numpy as np


def pca(X, var=0.95):
    """Return principal directions preserving at least var variance.

    Args:
        X: Centered data matrix of shape (n, d).
        var: Fraction of original variance to retain.

    Returns:
        Weights matrix W of shape (d, nd), where nd is the number of
        selected principal components.
    """
    _, singular_values, V_transpose = np.linalg.svd(X, full_matrices=False)
    cumulative_variance = np.cumsum(singular_values)
    components = np.searchsorted(
        cumulative_variance / cumulative_variance[-1], var
    ) + 1

    return V_transpose[:components].T
