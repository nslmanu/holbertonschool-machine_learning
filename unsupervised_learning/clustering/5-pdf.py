#!/usr/bin/env python3
"""Fonction de densité d'une distribution gaussienne multivariée."""

import numpy as np


def pdf(X, m, S):
    """Évalue la densité de probabilité gaussienne pour les points de ``X``.

    Args:
        X (numpy.ndarray): Points à évaluer, de forme (n, d).
        m (numpy.ndarray): Moyenne de la gaussienne, de forme (d,).
        S (numpy.ndarray): Matrice de covariance, de forme (d, d).

    Returns:
        numpy.ndarray: Densités de forme (n,), ou ``None`` si les arguments
        ne décrivent pas une distribution valide.
    """
    if type(X) is not np.ndarray or type(m) is not np.ndarray:
        return None
    if type(S) is not np.ndarray:
        return None
    if len(X.shape) != 2 or len(m.shape) != 1 or len(S.shape) != 2:
        return None

    _, d = X.shape
    if m.shape != (d,) or S.shape != (d, d):
        return None

    determinant = np.linalg.det(S)
    if determinant <= 0:
        return None

    try:
        inverse = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return None

    centered = X - m
    exponent = -0.5 * np.sum((centered @ inverse) * centered, axis=1)
    normalizer = np.sqrt(((2 * np.pi) ** d) * determinant)
    P = np.exp(exponent) / normalizer

    return np.maximum(P, 1e-300)
