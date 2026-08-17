#!/usr/bin/env python3
"""Module that calculates the sum of i squared."""


def summation_i_squared(n):
    """Calculate sum of i squared from 1 to n.

    Return integer value, or None if n is not valid.
    """
    if not isinstance(n, (int, float)) or n < 1:
        return None
    return int(n * (n + 1) * (2 * n + 1) / 6)
