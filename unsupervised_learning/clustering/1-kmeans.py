#!/usr/bin/env python3
"""Implémentation de l'algorithme K-means."""

import numpy as np

initialize = __import__('0-initialize').initialize


def kmeans(X, k, iterations=1000):
    """Regroupe les observations de ``X`` en ``k`` clusters.

    Args:
        X (numpy.ndarray): Jeu de données de forme (n, d).
        k (int): Nombre de clusters à construire.
        iterations (int): Nombre maximal de mises à jour des centroïdes.

    Returns:
        tuple: Les centroïdes de forme (k, d) et les indices de cluster de
        forme (n,), ou ``(None, None)`` si les arguments sont invalides.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    centroids = initialize(X, k)
    if centroids is None:
        return None, None

    for _ in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        clss = np.argmin(distances, axis=1)
        previous_centroids = centroids.copy()

        for cluster in range(k):
            points = X[clss == cluster]
            if points.size == 0:
                centroids[cluster] = np.random.uniform(
                    low=np.min(X, axis=0),
                    high=np.max(X, axis=0),
                )
            else:
                centroids[cluster] = np.mean(points, axis=0)

        if np.array_equal(centroids, previous_centroids):
            return centroids, clss

    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    clss = np.argmin(distances, axis=1)
    return centroids, clss
