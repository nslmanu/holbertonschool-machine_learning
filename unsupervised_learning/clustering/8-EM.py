#!/usr/bin/env python3
"""Algorithme Expectation-Maximization pour un GMM."""

import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5,
                             verbose=False):
    """Ajuste un Gaussian Mixture Model aux données ``X`` avec EM.

    Args:
        X (numpy.ndarray): Données de forme (n, d).
        k (int): Nombre de composantes gaussiennes.
        iterations (int): Nombre maximal d'itérations.
        tol (float): Seuil d'arrêt sur l'évolution de la log-vraisemblance.
        verbose (bool): Indique s'il faut afficher la progression.

    Returns:
        tuple: ``(pi, m, S, g, log_likelihood)`` ou cinq ``None`` si les
        paramètres sont invalides.
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None, None
    if type(k) is not int or k <= 0:
        return None, None, None, None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None, None
    if type(tol) is not float or tol < 0:
        return None, None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None:
        return None, None, None, None, None

    g, log_likelihood = expectation(X, pi, m, S)
    if g is None:
        return None, None, None, None, None

    if verbose:
        print("Log Likelihood after 0 iterations: {}".format(
            round(log_likelihood, 5)))

    for iteration in range(1, iterations + 1):
        pi, m, S = maximization(X, g)
        if pi is None:
            return None, None, None, None, None

        g, new_log_likelihood = expectation(X, pi, m, S)
        if g is None:
            return None, None, None, None, None

        difference = abs(new_log_likelihood - log_likelihood)
        log_likelihood = new_log_likelihood

        if verbose and iteration % 10 == 0:
            print("Log Likelihood after {} iterations: {}".format(
                iteration, round(log_likelihood, 5)))

        if difference <= tol:
            break

    if verbose and iteration % 10 != 0:
        print("Log Likelihood after {} iterations: {}".format(
            iteration, round(log_likelihood, 5)))

    return pi, m, S, g, log_likelihood
