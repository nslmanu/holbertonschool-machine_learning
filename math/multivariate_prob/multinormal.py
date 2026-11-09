#!/usr/bin/env python3
"""Represent a Multivariate Normal distribution."""
import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution."""

    def __init__(self, data):
        """Initializes a MultiNormal instance.

        Args:
            data (numpy.ndarray): shape (d, n), the data set with n data
                points of d dimensions.

        Raises:
            TypeError: if data is not a 2D numpy.ndarray.
            ValueError: if data contains fewer than 2 data points.
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        deviation = data - self.mean
        self.cov = np.matmul(deviation, deviation.T) / (n - 1)
