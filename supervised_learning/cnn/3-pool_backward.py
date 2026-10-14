#!/usr/bin/env python3
"""Back propagation over a pooling layer."""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs back propagation over a pooling layer.

    Args:
        dA (numpy.ndarray): shape (m, h_new, w_new, c), gradient with
            respect to the output of the pooling layer.
        A_prev (numpy.ndarray): shape (m, h_prev, w_prev, c), output of
            the previous layer.
        kernel_shape (tuple): (kh, kw), the pooling window size.
        stride (tuple): (sh, sw), the strides.
        mode (str): 'max' or 'avg'.

    Returns:
        numpy.ndarray: the gradient with respect to the previous layer.
    """
    m, h_new, w_new, c = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)
    for img in range(m):
        for i in range(h_new):
            for j in range(w_new):
                for k in range(c):
                    vs, hs = i * sh, j * sw
                    grad = dA[img, i, j, k]
                    if mode == 'max':
                        window = A_prev[img, vs:vs + kh, hs:hs + kw, k]
                        mask = (window == np.max(window))
                        dA_prev[img, vs:vs + kh, hs:hs + kw, k] += mask * grad
                    else:
                        avg = grad / (kh * kw)
                        dA_prev[img, vs:vs + kh, hs:hs + kw, k] += \
                            np.ones((kh, kw)) * avg
    return dA_prev
