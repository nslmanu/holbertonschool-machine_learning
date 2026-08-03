#!/usr/bin/env python3
"""def add_matrices2D(mat1, mat2): that adds two matrices element-wise:"""


def add_matrices2D(mat1, mat2):
    """Add arr1 and arr2 element-wise.

    Return a new list, or None if shapes size differ.
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    result = []
    for i in range(len(mat1)):
        new_line = []
        for j in range(len(mat1[0])):
            new_line.append(mat1[i][j] + mat2[i][j])
        result.append(new_line)
    return result
