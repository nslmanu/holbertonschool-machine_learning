#!/usr/bin/env python3
"""Compute binomial likelihoods for several candidate probabilities."""

import numpy as np


def likelihood(x, n, p):
    """Calculate likelihoods for observed binomial data."""
    if type(n) is not int or n <= 0:
        raise ValueError('n must be a positive integer')
    if type(x) is not int or x < 0:
        raise ValueError(
            'x must be an integer that is greater than or equal to 0'
        )
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(p, np.ndarray) or p.ndim != 1:
        raise TypeError('p must be a 1D numpy.ndarray')
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError(
            'p must be a 1D numpy.ndarray with values in the range [0, 1]'
        )

    combinations = 1
    for value in range(1, min(x, n - x) + 1):
        combinations *= (n - value + 1) / value

    return combinations * p ** x * (1 - p) ** (n - x)
