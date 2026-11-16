#!/usr/bin/env python3
"""Compute binomial likelihoods for candidate probabilities."""

import numpy as np


def likelihood(x, n, P):
    """Calculate likelihoods for observed binomial data.

    Args:
        x: Number of observed successes.
        n: Total number of trials.
        P: One-dimensional array of candidate success probabilities.

    Returns:
        A one-dimensional array of likelihoods corresponding to P.
    """
    if type(n) is not int or n <= 0:
        raise ValueError('n must be a positive integer')
    if type(x) is not int or x < 0:
        raise ValueError(
            'x must be an integer that is greater than or equal to 0'
        )
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError(
            'All values in P must be in the range [0, 1]'
        )

    combinations = 1
    for value in range(1, min(x, n - x) + 1):
        combinations *= (n - value + 1) / value

    return combinations * P ** x * (1 - P) ** (n - x)
