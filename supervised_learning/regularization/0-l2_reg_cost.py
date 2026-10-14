#!/usr/bin/env python3
"""Module that calculates the cost of a network with L2 regularization."""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculate the cost of a neural network with L2 regularization.

    Args:
        cost (float): the cost of the network without L2 regularization.
        lambtha (float): the regularization parameter.
        weights (dict): the weights and biases of the neural network.
        L (int): the number of layers in the neural network.
        m (int): the number of data points used.

    Returns:
        float: the cost of the network accounting for L2 regularization.
    """
    l2_sum = 0
    for i in range(1, L + 1):
        l2_sum += np.sum(np.square(weights['W' + str(i)]))
    return cost + (lambtha / (2 * m)) * l2_sum
