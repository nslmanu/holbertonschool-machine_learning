#!/usr/bin/env python3
"""Étape de maximisation de l'algorithme EM pour un GMM."""

import numpy as np


def maximization(X, g):
    """Met à jour les paramètres d'un GMM depuis les probabilités ``g``.

    Args:
        X (numpy.ndarray): Données de forme (n, d).
        g (numpy.ndarray): Probabilités a posteriori de forme (k, n).

    Returns:
        tuple: ``(pi, m, S)`` ou ``(None, None, None)`` si les entrées sont
        invalides.
    """
    if type(X) is not np.ndarray or type(g) is not np.ndarray:
        return None, None, None
    if len(X.shape) != 2 or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape
    k = g.shape[0]
    if n == 0 or k == 0 or g.shape[1] != n or np.any(g < 0):
        return None, None, None

    weights = np.sum(g, axis=1)
    if np.any(weights == 0):
        return None, None, None

    pi = weights / n
    m = (g @ X) / weights[:, np.newaxis]
    S = np.empty((k, d, d))

    for cluster in range(k):
        centered = X - m[cluster]
        S[cluster] = ((g[cluster, :, np.newaxis] * centered).T @ centered /
                      weights[cluster])

    return pi, m, S
