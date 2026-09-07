#!/usr/bin/env python3
"""Module that updates a variable using the RMSProp algorithm."""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Update a variable using the RMSProp optimization algorithm.

    Args:
        alpha (float): the learning rate.
        beta2 (float): the RMSProp weight.
        epsilon (float): a small number to avoid division by zero.
        var (numpy.ndarray): the variable to be updated.
        grad (numpy.ndarray): the gradient of var.
        s (numpy.ndarray): the previous second moment of var.

    Returns:
        tuple: the updated variable and the new moment, respectively.
    """
    s = beta2 * s + (1 - beta2) * (grad ** 2)
    var = var - alpha * grad / (s ** 0.5 + epsilon)
    return var, s
