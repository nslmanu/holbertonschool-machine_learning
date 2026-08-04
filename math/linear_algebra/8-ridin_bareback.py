#!/usr/bin/env python3
"""mat_mul(mat1, mat2): that performs matrix multiplication:"""


def mat_mul(mat1, mat2):
    """mult arr1 and arr2 to new element-wise.

    Return a new list. hori and vert
    """
    if len(mat1[0]) != len(mat2):
        return None

    result = []

    for each_lines in range(len(mat1)):
        new_line = []
        for each_col in range(len(mat2[0])):
            total = 0
            for each_linesBIS in range(len(mat2)):
                total += mat1[each_lines][each_linesBIS] * mat2[each_linesBIS][each_col]
            new_line.append(total)
        result.append(new_line)
    return result
