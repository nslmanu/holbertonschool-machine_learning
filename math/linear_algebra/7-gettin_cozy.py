#!/usr/bin/env python3
"""def cat_matrices2D(mat1, mat2, axis=0): that concatenates
two matrices along a specific axis:"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Add arr1 and arr2 to new element-wise.

    Return a new list. hori and vert
    """

    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None

        result = []
        for row in mat1:
            result.append(row[:])
        for row in mat2:
            result.append(row[:])
        return result

    if axis == 1:
        if len(mat1) != len(mat2):
            return None

        result = []
        for i in range(len(mat1)):
            new_row = []
            for j in range(len(mat1[i])):
                new_row.append(mat1[i][j])
            for j in range(len(mat2[i])):
                new_row.append(mat2[i][j])
            result.append(new_row)
        return result

    return None
