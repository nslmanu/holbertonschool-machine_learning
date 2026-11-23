#!/usr/bin/env python3
"""K-means avec scikit-learn."""

import sklearn.cluster


def kmeans(X, k):
    """Regroupe les données de ``X`` en ``k`` clusters avec K-means.

    Args:
        X (numpy.ndarray): Données de forme (n, d).
        k (int): Nombre de clusters.

    Returns:
        tuple: Les centroïdes de forme (k, d) et les labels de forme (n,).
    """
    model = sklearn.cluster.KMeans(n_clusters=k)
    model.fit(X)
    return model.cluster_centers_, model.labels_
