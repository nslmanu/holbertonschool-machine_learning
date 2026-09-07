#!/usr/bin/env python3
"""Module that updates the learning rate using inverse time decay."""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Update the learning rate using inverse time decay (stepwise).

    Args:
        alpha (float): the original learning rate.
        decay_rate (float): the weight used to determine the decay rate.
        global_step (int): the number of gradient descent passes elapsed.
        decay_step (int): the number of passes before alpha decays further.

    Returns:
        float: the updated value for alpha.
    """
    alpha = alpha / (1 + decay_rate * (global_step // decay_step))
    return alpha
