#!/usr/bin/env python3
"""Calculate the marginal probability of observed binomial data."""

import numpy as np


def marginal(x, n, P, Pr):
    """Return the probability of the data across all hypotheses.

    The result is the sum of the intersections of the data with every
    candidate probability hypothesis.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError('n must be a positive integer')
    if not isinstance(x, int) or x < 0:
        raise ValueError('x must be an integer that is greater than or equal to 0')
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError('Pr must be a numpy.ndarray with the same shape as P')
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError('All values in P must be in the range [0, 1]')
    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError('All values in Pr must be in the range [0, 1]')
    if not np.isclose(Pr.sum(), 1):
        raise ValueError('Pr must sum to 1')

    combinations = 1
    for value in range(1, min(x, n - x) + 1):
        combinations *= (n - value + 1) / value

    likelihood = combinations * P ** x * (1 - P) ** (n - x)
    return np.sum(likelihood * Pr)
