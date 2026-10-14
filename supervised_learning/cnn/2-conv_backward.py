#!/usr/bin/env python3
"""Back propagation over a convolutional layer."""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Performs back propagation over a convolutional layer.

    Args:
        dZ (numpy.ndarray): shape (m, h_new, w_new, c_new), gradient
            with respect to the unactivated output of the layer.
        A_prev (numpy.ndarray): shape (m, h_prev, w_prev, c_prev),
            output of the previous layer.
        W (numpy.ndarray): shape (kh, kw, c_prev, c_new), the kernels.
        b (numpy.ndarray): shape (1, 1, 1, c_new), the biases.
        padding (str): 'same' or 'valid'.
        stride (tuple): (sh, sw), the strides.

    Returns:
        tuple: (dA_prev, dW, db), gradients w.r.t. the previous layer,
        the kernels, and the biases.
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph, pw = 0, 0

    A_pad = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                   mode='constant')
    dA_pad = np.zeros_like(A_pad)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                vs, hs = i * sh, j * sw
                window = A_pad[:, vs:vs + kh, hs:hs + kw, :]
                grad = dZ[:, i, j, k][:, None, None, None]
                dA_pad[:, vs:vs + kh, hs:hs + kw, :] += W[:, :, :, k] * grad
                dW[:, :, :, k] += np.sum(window * grad, axis=0)

    if padding == 'same':
        dA_prev = dA_pad[:, ph:ph + h_prev, pw:pw + w_prev, :]
    else:
        dA_prev = dA_pad
    return dA_prev, dW, db
