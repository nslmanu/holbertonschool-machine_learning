#!/usr/bin/env python3
"""Module that conducts forward propagation using Dropout."""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Conduct forward propagation using Dropout.

    All layers except the last use the tanh activation function; the last
    layer uses the softmax activation function.

    Args:
        X (numpy.ndarray): input data of shape (nx, m).
        weights (dict): the weights and biases of the neural network.
        L (int): the number of layers in the network.
        keep_prob (float): the probability that a node will be kept.

    Returns:
        dict: the outputs of each layer and the dropout mask used on each
            layer.
    """
    cache = {}
    cache['A0'] = X

    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A_prev = cache['A' + str(i - 1)]
        Z = np.matmul(W, A_prev) + b

        if i == L:
            t = np.exp(Z)
            cache['A' + str(i)] = t / np.sum(t, axis=0, keepdims=True)
        else:
            A = np.tanh(Z)
            D = np.random.binomial(1, keep_prob, size=A.shape)
            A = (A * D) / keep_prob
            cache['A' + str(i)] = A
            cache['D' + str(i)] = D

    return cache
