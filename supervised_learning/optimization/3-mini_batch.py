#!/usr/bin/env python3
"""Module that creates mini-batches for mini-batch gradient descent."""
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """Create mini-batches to be used for training a neural network.

    Args:
        X (numpy.ndarray): input data of shape (m, nx).
        Y (numpy.ndarray): labels of shape (m, ny).
        batch_size (int): the number of data points in a batch.

    Returns:
        list: mini-batches, each a tuple (X_batch, Y_batch).
    """
    m = X.shape[0]
    X_shuffled, Y_shuffled = shuffle_data(X, Y)

    mini_batches = []
    for i in range(0, m, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        Y_batch = Y_shuffled[i:i + batch_size]
        mini_batches.append((X_batch, Y_batch))
    return mini_batches
