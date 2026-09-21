#!/usr/bin/env python3
"""Forward propagation over a pooling layer."""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs forward propagation over a pooling layer.

    Args:
        A_prev (numpy.ndarray): shape (m, h_prev, w_prev, c_prev),
            output of the previous layer.
        kernel_shape (tuple): (kh, kw), the pooling window size.
        stride (tuple): (sh, sw), the strides.
        mode (str): 'max' or 'avg'.

    Returns:
        numpy.ndarray: the output of the pooling layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    oh = (h_prev - kh) // sh + 1
    ow = (w_prev - kw) // sw + 1
    output = np.zeros((m, oh, ow, c_prev))
    op = np.max if mode == 'max' else np.mean
    for i in range(oh):
        for j in range(ow):
            window = A_prev[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
            output[:, i, j, :] = op(window, axis=(1, 2))
    return output
