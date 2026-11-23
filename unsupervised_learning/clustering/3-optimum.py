#!/usr/bin/env python3
"""Calcul de la variance intra-cluster totale."""

import numpy as np


def variance(X, C):
    """Calcule la somme des distances au carré au centroïde le plus proche.

    Args:
        X (numpy.ndarray): Données de forme (n, d).
        C (numpy.ndarray): Centroïdes de forme (k, d).

    Returns:
        float: Variance intra-cluster totale, ou ``None`` en cas d'erreur.
    """
    if not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray):
        return None
    if X.ndim != 2 or C.ndim != 2 or X.shape[1] != C.shape[1]:
        return None
    if X.shape[0] == 0 or C.shape[0] == 0:
        return None

    distances_squared = np.sum((X[:, np.newaxis] - C) ** 2, axis=2)
    return np.sum(np.min(distances_squared, axis=1))
