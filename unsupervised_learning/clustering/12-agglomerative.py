#!/usr/bin/env python3
"""Clustering hiérarchique agglomératif avec Ward linkage."""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Regroupe ``X`` par clustering agglomératif et affiche le dendrogramme.

    Args:
        X (numpy.ndarray): Données de forme (n, d).
        dist (float): Distance maximale de coupure entre les clusters.

    Returns:
        numpy.ndarray: Labels de cluster de forme (n,).
    """
    linkage_matrix = scipy.cluster.hierarchy.linkage(X, method='ward')
    clss = scipy.cluster.hierarchy.fcluster(
        linkage_matrix, t=dist, criterion='distance')

    scipy.cluster.hierarchy.dendrogram(
        linkage_matrix, color_threshold=dist, above_threshold_color='b')
    plt.show()

    return clss
