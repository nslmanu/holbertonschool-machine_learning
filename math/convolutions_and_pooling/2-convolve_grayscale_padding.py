#!/usr/bin/env python3
"""Convolution on grayscale images with custom padding."""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Performs a convolution on grayscale images with custom padding.

    Args:
        images (numpy.ndarray): shape (m, h, w) with m grayscale images.
        kernel (numpy.ndarray): shape (kh, kw), the convolution kernel.
        padding (tuple): (ph, pw), zero-padding for height and width.

    Returns:
        numpy.ndarray of shape
        (m, h + 2*ph - kh + 1, w + 2*pw - kw + 1): the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')
    oh = h + 2 * ph - kh + 1
    ow = w + 2 * pw - kw + 1
    output = np.zeros((m, oh, ow))
    for i in range(oh):
        for j in range(ow):
            window = padded[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2))
    return output
