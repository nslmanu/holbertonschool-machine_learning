#!/usr/bin/env python3
"""Calculate the minor matrix of a matrix."""
determinant = __import__('0-determinant').determinant


def minor(matrix):
    """Calculates the minor matrix of a matrix.

    Args:
        matrix (list of lists): the matrix whose minor matrix to compute.

    Returns:
        list of lists: the minor matrix of matrix.

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

    if n == 1:
        return [[1]]

    minors = []
    for i in range(n):
        row = []
        for j in range(n):
            sub = [[matrix[r][c] for c in range(n) if c != j]
                   for r in range(n) if r != i]
            row.append(determinant(sub))
        minors.append(row)
    return minors
