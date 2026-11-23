#!/usr/bin/env python3
"""Gaussian Mixture Model avec scikit-learn."""

import sklearn.mixture


def gmm(X, k):
    """Ajuste un GMM de ``k`` composantes aux données ``X``.

    Args:
        X (numpy.ndarray): Jeu de données de forme (n, d).
        k (int): Nombre de composantes gaussiennes.

    Returns:
        tuple: Les poids, moyennes, covariances, labels et BIC du modèle.
    """
    model = sklearn.mixture.GaussianMixture(n_components=k)
    model.fit(X)
    clss = model.predict(X)

    return (model.weights_, model.means_, model.covariances_, clss,
            model.bic(X))
