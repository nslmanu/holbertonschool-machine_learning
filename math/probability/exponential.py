#!/usr/bin/env python3
"""Represent an exponential distribution."""


class Exponential:
    """Store the rate governing exponentially distributed waiting times."""

    def __init__(self, data=None, lambtha=1.):
        """Initialize the rate directly or estimate it from waiting times."""
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = float(len(data) / sum(data))
