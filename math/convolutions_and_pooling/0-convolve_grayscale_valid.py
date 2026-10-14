#!/usr/bin/env python3
"""Valid convolution on grayscale images."""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Performs a valid convolution on grayscale images.

    Args:
        images (numpy.ndarray): shape (m, h, w) with m grayscale images.
        kernel (numpy.ndarray): shape (kh, kw), the convolution kernel.

    Returns:
        numpy.ndarray of shape (m, h - kh + 1, w - kw + 1): convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    oh = h - kh + 1
    ow = w - kw + 1
    output = np.zeros((m, oh, ow))
    for i in range(oh):
        for j in range(ow):
            window = images[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2))
    return output
