#!/usr/bin/env python3
"""Same convolution on grayscale images."""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs a same convolution on grayscale images.

    The output keeps the same height and width as the input; the image
    is zero-padded as needed so the kernel stays centered.

    Args:
        images (numpy.ndarray): shape (m, h, w) with m grayscale images.
        kernel (numpy.ndarray): shape (kh, kw), the convolution kernel.

    Returns:
        numpy.ndarray of shape (m, h, w): the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph = kh // 2
    pw = kw // 2
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')
    output = np.zeros((m, h, w))
    for i in range(h):
        for j in range(w):
            window = padded[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2))
    return output
