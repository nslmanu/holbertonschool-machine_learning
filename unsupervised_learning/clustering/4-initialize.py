#!/usr/bin/env python3
"""Initialisation des paramètres d'un Gaussian Mixture Model."""

import numpy as np

kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initialise les probabilités, moyennes et covariances d'un GMM.

    Args:
        X (numpy.ndarray): Jeu de données de forme (n, d).
        k (int): Nombre de composantes gaussiennes.

    Returns:
        tuple: ``(pi, m, S)`` ou ``(None, None, None)`` si les entrées sont
        invalides.
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(k) is not int or k <= 0:
        return None, None, None

    pi = np.full(k, 1 / k)
    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None
    S = np.tile(np.eye(X.shape[1]), (k, 1, 1))

    return pi, m, S
