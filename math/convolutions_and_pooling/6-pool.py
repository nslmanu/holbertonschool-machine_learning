#!/usr/bin/env python3
"""Pooling on images."""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Performs pooling on images.

    Args:
        images (numpy.ndarray): shape (m, h, w, c) with m images.
        kernel_shape (tuple): (kh, kw), the pooling window size.
        stride (tuple): (sh, sw), the stride for height and width.
        mode (str): 'max' for max pooling, 'avg' for average pooling.

    Returns:
        numpy.ndarray of shape (m, oh, ow, c): the pooled images.
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    oh = (h - kh) // sh + 1
    ow = (w - kw) // sw + 1
    output = np.zeros((m, oh, ow, c))
    op = np.max if mode == 'max' else np.mean
    for i in range(oh):
        for j in range(ow):
            window = images[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
            output[:, i, j, :] = op(window, axis=(1, 2))
    return output
