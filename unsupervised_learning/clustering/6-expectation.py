#!/usr/bin/env python3
"""Étape d'espérance de l'algorithme EM pour un GMM."""

import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Calcule les probabilités a posteriori et la log-vraisemblance.

    Args:
        X (numpy.ndarray): Données de forme (n, d).
        pi (numpy.ndarray): Probabilités a priori, de forme (k,).
        m (numpy.ndarray): Moyennes, de forme (k, d).
        S (numpy.ndarray): Covariances, de forme (k, d, d).

    Returns:
        tuple: ``(g, l)`` avec ``g`` de forme (k, n) et ``l`` la
        log-vraisemblance, ou ``(None, None)`` si les paramètres sont
        invalides.
    """
    if (type(X) is not np.ndarray or type(pi) is not np.ndarray or
            type(m) is not np.ndarray or type(S) is not np.ndarray):
        return None, None
    if (len(X.shape) != 2 or len(pi.shape) != 1 or len(m.shape) != 2 or
            len(S.shape) != 3):
        return None, None

    n, d = X.shape
    k = pi.shape[0]
    if (m.shape != (k, d) or S.shape != (k, d, d) or k == 0 or n == 0):
        return None, None
    if np.any(pi < 0) or not np.isclose(np.sum(pi), 1):
        return None, None

    g = np.empty((k, n))
    for cluster in range(k):
        density = pdf(X, m[cluster], S[cluster])
        if density is None:
            return None, None
        g[cluster] = pi[cluster] * density

    likelihoods = np.sum(g, axis=0)
    if np.any(likelihoods <= 0):
        return None, None

    g /= likelihoods
    log_likelihood = np.sum(np.log(likelihoods))
    return g, log_likelihood
