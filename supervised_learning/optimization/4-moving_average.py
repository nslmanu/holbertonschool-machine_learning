#!/usr/bin/env python3
"""Module that calculates the weighted moving average of a data set."""


def moving_average(data, beta):
    """Calculate the weighted moving average of a data set.

    Uses bias correction.

    Args:
        data (list): the list of data to calculate the moving average of.
        beta (float): the weight used for the moving average.

    Returns:
        list: the moving averages of data.
    """
    averages = []
    v = 0
    for i in range(len(data)):
        v = beta * v + (1 - beta) * data[i]
        bias_correction = v / (1 - beta ** (i + 1))
        averages.append(bias_correction)
    return averages
