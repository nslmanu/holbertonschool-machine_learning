#!/usr/bin/env python3
"""Initialisation des centroïdes pour l'algorithme K-means."""

import numpy as np


def initialize(X, k):
    """Initialise ``k`` centroïdes aléatoires dans les bornes de ``X``.

    Args:
        X (numpy.ndarray): Jeu de données de forme (n, d).
        k (int): Nombre de centroïdes à créer.

    Returns:
        numpy.ndarray: Centroïdes de forme (k, d), ou ``None`` si les
            arguments ne sont pas valides.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    return np.random.uniform(
        low=np.min(X, axis=0),
        high=np.max(X, axis=0),
        size=(k, X.shape[1]),
    )
