#!/usr/bin/env python3
"""Represent a binomial distribution."""


class Binomial:
    """Store the number of trials and probability of success."""

    def __init__(self, data=None, n=1, p=0.5):
        """Initialize parameters directly or estimate them from data."""
        if data is None:
            if n <= 0:
                raise ValueError('n must be a positive value')
            if not 0 < p < 1:
                raise ValueError('p must be greater than 0 and less than 1')
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            mean = sum(data) / len(data)
            variance = sum((value - mean) ** 2 for value in data) / len(data)
            estimated_p = 1 - variance / mean
            self.n = round(mean / estimated_p)
            self.p = float(mean / self.n)
