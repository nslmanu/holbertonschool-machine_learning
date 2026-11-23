#!/usr/bin/env python3
"""Sélection du nombre de composantes d'un GMM avec le BIC."""

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Trouve le meilleur nombre de clusters selon le BIC.

    Args:
        X (numpy.ndarray): Données de forme (n, d).
        kmin (int): Nombre minimal de clusters testé.
        kmax (int): Nombre maximal de clusters testé.
        iterations (int): Nombre maximal d'itérations d'EM.
        tol (float): Tolérance de convergence d'EM.
        verbose (bool): Indique s'il faut afficher l'avancement d'EM.

    Returns:
        tuple: ``(best_k, best_result, l, b)``, ou quatre ``None`` si les
        paramètres sont invalides.
    """
    if type(X) is not np.ndarray or len(X.shape) != 2 or X.shape[0] == 0:
        return None, None, None, None
    if type(kmin) is not int or kmin <= 0:
        return None, None, None, None

    if kmax is None:
        kmax = X.shape[0]

    if type(kmax) is not int or kmax <= 0 or kmax < kmin:
        return None, None, None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None
    if type(tol) is not float or tol < 0:
        return None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None

    n, d = X.shape
    l = np.empty(kmax - kmin + 1)
    b = np.empty(kmax - kmin + 1)
    best_k = None
    best_result = None
    lowest_bic = np.inf

    for index, k in enumerate(range(kmin, kmax + 1)):
        pi, m, S, _, log_likelihood = expectation_maximization(
            X, k, iterations, tol, verbose)
        if pi is None:
            return None, None, None, None

        parameter_count = (k * d * (d + 1) / 2) + (k * d) + (k - 1)
        l[index] = log_likelihood
        b[index] = (parameter_count * np.log(n)) - (2 * log_likelihood)

        if b[index] < lowest_bic:
            lowest_bic = b[index]
            best_k = k
            best_result = (pi, m, S)

    return best_k, best_result, l, b
