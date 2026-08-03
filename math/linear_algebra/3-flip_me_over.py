#!/usr/bin/env python3
"""Write a function def matrix_transpose(matrix):
 that returns each X of a 2D matrix, mat:"""


def matrix_transpose(datamat):
    """take 1st of each and insert and new, etc etc"""
    result = []
    for j in range(len(datamat[0])):
        new_row = []
        for i in range(len(datamat)):
            new_row.append(datamat[i][j])
        result.append(new_row)
    return result
