#!/usr/bin/env python3
"""Module qui calcule la shape d'une matrice."""


def matrix_shape(datamat):
    """Retourne la shape d'une matrice sous forme de liste d'entiers."""
    shape = []
    while isinstance(datamat, list):
        shape.append(len(datamat))
        datamat = datamat[0]
    return shape
