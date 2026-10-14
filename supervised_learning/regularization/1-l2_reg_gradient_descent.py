#!/usr/bin/env python3
"""Module that updates weights using gradient descent with L2 reg."""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Update the weights and biases using gradient descent with L2 reg.

    The network uses tanh activations on each layer except the last,
    which uses a softmax activation. Weights and biases are updated in
    place.

    Args:
        Y (numpy.ndarray): one-hot of shape (classes, m) with the correct
            labels for the data.
        weights (dict): the weights and biases of the neural network.
        cache (dict): the outputs of each layer of the neural network.
        alpha (float): the learning rate.
        lambtha (float): the L2 regularization parameter.
        L (int): the number of layers of the network.
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            dZ = np.matmul(W.T, dZ) * (1 - cache['A' + str(i - 1)] ** 2)

        weights['W' + str(i)] = W - alpha * dW
        weights['b' + str(i)] = weights['b' + str(i)] - alpha * db
