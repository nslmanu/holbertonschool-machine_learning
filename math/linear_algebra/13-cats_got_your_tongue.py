#!/usr/bin/env python3
"""concatenate on numpy ndarrays."""


def np_elementwise(mat1, mat2):
    """array of sum, difference, product, quotient."""
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
