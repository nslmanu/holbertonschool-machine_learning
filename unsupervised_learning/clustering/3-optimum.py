#!/usr/bin/env python3
"""Find the optimum number of clusters."""

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Test K-means with different cluster sizes."""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None

    if type(kmin) is not int or kmin <= 0:
        return None, None

    if kmax is None:
        kmax = kmin

    if type(kmax) is not int or kmax <= 0 or kmax <= kmin:
        return None, None

    if type(iterations) is not int or iterations <= 0:
        return None, None

    results = []
    d_vars = []
    first_variance = None

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))

        current_variance = variance(X, C)

        if first_variance is None:
            first_variance = current_variance

        d_vars.append(first_variance - current_variance)

    return results, d_vars