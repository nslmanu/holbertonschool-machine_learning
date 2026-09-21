#!/usr/bin/env python3
"""Forward propagation over a convolutional layer."""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Performs forward propagation over a convolutional layer.

    Args:
        A_prev (numpy.ndarray): shape (m, h_prev, w_prev, c_prev),
            output of the previous layer.
        W (numpy.ndarray): shape (kh, kw, c_prev, c_new), the kernels.
        b (numpy.ndarray): shape (1, 1, 1, c_new), the biases.
        activation (function): activation applied to the convolution.
        padding (str): 'same' or 'valid'.
        stride (tuple): (sh, sw), the strides.

    Returns:
        numpy.ndarray: the output of the convolutional layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph, pw = 0, 0

    A_pad = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                   mode='constant')
    oh = (h_prev + 2 * ph - kh) // sh + 1
    ow = (w_prev + 2 * pw - kw) // sw + 1
    Z = np.zeros((m, oh, ow, c_new))
    for i in range(oh):
        for j in range(ow):
            window = A_pad[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
            Z[:, i, j, :] = np.sum(window[..., np.newaxis] * W,
                                   axis=(1, 2, 3))
    return activation(Z + b)
