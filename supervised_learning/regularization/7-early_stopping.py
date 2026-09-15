#!/usr/bin/env python3
"""Module that determines if you should stop gradient descent early."""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Determine if you should stop gradient descent early.

    Early stopping occurs when the validation cost has not decreased
    relative to the optimal validation cost by more than the threshold
    over a specific patience count.

    Args:
        cost (float): the current validation cost of the network.
        opt_cost (float): the lowest recorded validation cost.
        threshold (float): the threshold used for early stopping.
        patience (int): the patience count used for early stopping.
        count (int): the count of how long the threshold has not been met.

    Returns:
        tuple: a boolean of whether the network should be stopped early,
            followed by the updated count.
    """
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1

    if count >= patience:
        return True, count
    return False, count
