#!/usr/bin/env python3
"""Fonction permettant de trouver le k optimal."""

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Teste plusieurs nombres de clusters avec K-means."""
    if kmax is None:
        kmax = kmin

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    if not isinstance(kmin, int) or kmin <= 0:
        return None, None

    if not isinstance(kmax, int) or kmax <= 0:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    if kmax <= kmin:
        return None, None

    results = []
    d_vars = []
    first_variance = None

    for k in range(kmin, kmax + 1):
        centroids, clss = kmeans(X, k, iterations)
        current_variance = variance(X, centroids)

        results.append((centroids, clss))

        if first_variance is None:
            first_variance = current_variance

        d_vars.append(first_variance - current_variance)

    return results, d_vars
