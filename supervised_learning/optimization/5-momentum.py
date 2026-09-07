#!/usr/bin/env python3
"""Module that updates a variable using gradient descent with momentum."""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Update a variable using the gradient descent with momentum algorithm.

    Args:
        alpha (float): the learning rate.
        beta1 (float): the momentum weight.
        var (numpy.ndarray): the variable to be updated.
        grad (numpy.ndarray): the gradient of var.
        v (numpy.ndarray): the previous first moment of var.

    Returns:
        tuple: the updated variable and the new moment, respectively.
    """
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v
    return var, v
