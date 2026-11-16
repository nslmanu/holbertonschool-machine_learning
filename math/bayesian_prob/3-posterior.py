#!/usr/bin/env python3
"""Calculate posterior probabilities for Bayesian hypotheses."""

import numpy as np


def posterior(x, n, P, Pr):
    """Calculate the posterior probability for each hypothesis.

    Args:
        x: Number of observed successes.
        n: Total number of trials.
        P: One-dimensional array of candidate probabilities.
        Pr: One-dimensional array of probabilities a priori.

    Returns:
        A one-dimensional array of posterior probabilities.
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
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError(
            'Pr must be a numpy.ndarray with the same shape as P'
        )
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError(
            'All values in P must be in the range [0, 1]'
        )
    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError(
            'All values in Pr must be in the range [0, 1]'
        )
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError('Pr must sum to 1')

    combinations = 1
    for value in range(1, min(x, n - x) + 1):
        combinations *= (n - value + 1) / value

    intersections = combinations * P ** x * (1 - P) ** (n - x) * Pr
    marginal = np.sum(intersections)

    return intersections / marginal
