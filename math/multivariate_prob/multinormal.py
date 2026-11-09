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

    def pdf(self, x):
        """Calculates the PDF at a data point.

        Args:
            x (numpy.ndarray): shape (d, 1), the data point whose PDF
                should be calculated.

        Returns:
            float: the value of the PDF at x.

        Raises:
            TypeError: if x is not a numpy.ndarray.
            ValueError: if x does not have the shape (d, 1).
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.cov.shape[0]
        if x.ndim != 2 or x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        diff = x - self.mean
        exponent = -0.5 * np.matmul(np.matmul(diff.T, inv), diff)[0][0]
        coefficient = 1.0 / np.sqrt(((2 * np.pi) ** d) * det)
        return float(coefficient * np.exp(exponent))
