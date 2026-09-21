#!/usr/bin/env python3
"""Convolution on images using multiple kernels."""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Performs a convolution on images using multiple kernels.

    Args:
        images (numpy.ndarray): shape (m, h, w, c) with m images.
        kernels (numpy.ndarray): shape (kh, kw, c, nc), the kernels.
        padding (str or tuple): 'same', 'valid', or (ph, pw).
        stride (tuple): (sh, sw), the stride for height and width.

    Returns:
        numpy.ndarray of shape (m, oh, ow, nc): the convolved images.
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
    else:
        ph, pw = padding

    padded = np.pad(images,
                    ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    mode='constant')
    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1
    output = np.zeros((m, oh, ow, nc))
    for i in range(oh):
        for j in range(ow):
            for k in range(nc):
                window = padded[:, i * sh:i * sh + kh,
                                j * sw:j * sw + kw, :]
                output[:, i, j, k] = np.sum(window * kernels[:, :, :, k],
                                            axis=(1, 2, 3))
    return output
