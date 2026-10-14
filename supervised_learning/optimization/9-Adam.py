#!/usr/bin/env python3
"""Module that updates a variable using the Adam optimization algorithm."""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Update a variable in place using the Adam optimization algorithm.

    Args:
        alpha (float): the learning rate.
        beta1 (float): the weight used for the first moment.
        beta2 (float): the weight used for the second moment.
        epsilon (float): a small number to avoid division by zero.
        var (numpy.ndarray): the variable to be updated.
        grad (numpy.ndarray): the gradient of var.
        v (numpy.ndarray): the previous first moment of var.
        s (numpy.ndarray): the previous second moment of var.
        t (int): the time step used for bias correction.

    Returns:
        tuple: the updated variable, the new first moment, and the new
            second moment, respectively.
    """
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * (grad ** 2)
    v_corrected = v / (1 - beta1 ** t)
    s_corrected = s / (1 - beta2 ** t)
    var = var - alpha * v_corrected / (s_corrected ** 0.5 + epsilon)
    return var, v, s
