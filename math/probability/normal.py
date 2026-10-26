#!/usr/bin/env python3
"""Represent a normal distribution."""


class Normal:
    """Store the mean and standard deviation of a normal distribution."""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize parameters directly or estimate them from data."""
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = float(sum(data) / len(data))
            variance = sum((value - self.mean) ** 2 for value in data)
            self.stddev = float((variance / len(data)) ** 0.5)

    def z_score(self, x):
        """Return the number of standard deviations x is from the mean."""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Return the original value corresponding to a z-score."""
        return self.mean + z * self.stddev
