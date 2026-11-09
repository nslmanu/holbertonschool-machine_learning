#!/usr/bin/env python3
"""Calculate the cofactor matrix of a matrix."""
minor = __import__('1-minor').minor


def cofactor(matrix):
    """Calculates the cofactor matrix of a matrix.

    Args:
        matrix (list of lists): the matrix whose cofactor matrix to compute.

    Returns:
        list of lists: the cofactor matrix of matrix.

    Raises:
        TypeError: if matrix is not a list of lists.
        ValueError: if matrix is not a non-empty square matrix.
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    minors = minor(matrix)
    return [[((-1) ** (i + j)) * minors[i][j] for j in range(n)]
            for i in range(n)]
