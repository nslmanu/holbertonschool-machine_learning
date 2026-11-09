#!/usr/bin/env python3
"""Calculate the determinant of a matrix."""


def determinant(matrix):
    """Calculates the determinant of a matrix.

    Args:
        matrix (list of lists): the matrix whose determinant to compute.

    Returns:
        int/float: the determinant of matrix.

    Raises:
        TypeError: if matrix is not a list of lists.
        ValueError: if matrix is not square.
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # 0x0 matrix represented by [[]]
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1

    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for j in range(n):
        sub = [[matrix[i][c] for c in range(n) if c != j]
               for i in range(1, n)]
        det += ((-1) ** j) * matrix[0][j] * determinant(sub)
    return det
