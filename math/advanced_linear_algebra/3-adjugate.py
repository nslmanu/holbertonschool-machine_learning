#!/usr/bin/env python3
"""Calculate the adjugate matrix of a matrix."""
cofactor = __import__('2-cofactor').cofactor


def adjugate(matrix):
    """Calculates the adjugate matrix of a matrix.

    Args:
        matrix (list of lists): the matrix whose adjugate to compute.

    Returns:
        list of lists: the adjugate (transpose of the cofactor matrix).

    Raises:
        TypeError: if matrix is not a list of lists.
        ValueError: if matrix is not a non-empty square matrix.
    """
    cof = cofactor(matrix)
    n = len(cof)
    return [[cof[j][i] for j in range(n)] for i in range(n)]
