#!/usr/bin/env python3
"""Represent a Poisson distribution."""


class Poisson:
    """Store the expected number of occurrences in a given time frame."""

    def __init__(self, data=None, lambtha=1.):
        """Initialize the rate directly or estimate it from a data list."""
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Return the probability of exactly k occurrences."""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        factorial = 1
        for value in range(2, k + 1):
            factorial *= value

        e = 2.7182818285
        return (e ** (-self.lambtha) * self.lambtha ** k
                / factorial)
