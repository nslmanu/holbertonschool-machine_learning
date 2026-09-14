#!/usr/bin/env python3
"""Module that updates weights with Dropout using gradient descent."""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Update the weights of a network with Dropout using gradient descent.

    All layers use the tanh activation function except the last, which uses
    the softmax activation function. The weights are updated in place.

    Args:
        Y (numpy.ndarray): one-hot of shape (classes, m) with the correct
            labels for the data.
        weights (dict): the weights and biases of the neural network.
        cache (dict): the outputs and dropout masks of each layer.
        alpha (float): the learning rate.
        keep_prob (float): the probability that a node will be kept.
        L (int): the number of layers of the network.
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            dA = np.matmul(W.T, dZ)
            dA = dA * (1 - cache['A' + str(i - 1)] ** 2)
            dA = (dA * cache['D' + str(i - 1)]) / keep_prob
            dZ = dA

        weights['W' + str(i)] = W - alpha * dW
        weights['b' + str(i)] = weights['b' + str(i)] - alpha * db
