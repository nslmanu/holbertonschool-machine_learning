#!/usr/bin/env python3
"""Module that adds two arrays element."""


def add_arrays(arr1, arr2):
    """Add arr1 and arr2 element-wise.

    Return a new list, or None if shapes size differ.
    """
    if len(arr1) != len(arr2):
        return None

    result = []
    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])
    return result
