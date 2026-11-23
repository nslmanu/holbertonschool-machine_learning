#!/usr/bin/env python3
"""Recherche du nombre de clusters le plus pertinent par variance."""

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Exécute K-means pour plusieurs valeurs de ``k``.

    Args:
        X (numpy.ndarray): Données de forme (n, d).
        kmin (int): Plus petite valeur de k à tester.
        kmax (int): Plus grande valeur de k à tester.
        iterations (int): Nombre maximal d'itérations de K-means.

    Returns:
        tuple: Résultats de K-means et améliorations de variance, ou
        ``(None, None)`` si les paramètres sont invalides.
    """
    if kmax is None:
        kmax = kmin

    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] == 0:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if not isinstance(kmax, int) or kmax <= 0 or kmax <= kmin:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    results = []
    d_vars = []
    initial_variance = None

    for k in range(kmin, kmax + 1):
        centroids, clss = kmeans(X, k, iterations)
        current_variance = variance(X, centroids)
        results.append((centroids, clss))

        if initial_variance is None:
            initial_variance = current_variance
        d_vars.append(initial_variance - current_variance)

    return results, d_vars