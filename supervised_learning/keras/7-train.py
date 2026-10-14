#!/usr/bin/env python3
"""Module that trains a model using mini-batch gradient descent."""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """Train a model using mini-batch gradient descent.

    Args:
        network (keras.Model): the model to train.
        data (numpy.ndarray): input data of shape (m, nx).
        labels (numpy.ndarray): one-hot labels of shape (m, classes).
        batch_size (int): size of the batch for mini-batch gradient descent.
        epochs (int): number of passes through data.
        validation_data (tuple): the data to validate the model with.
        early_stopping (bool): whether to use early stopping.
        patience (int): the patience used for early stopping.
        learning_rate_decay (bool): whether to use learning rate decay.
        alpha (float): the initial learning rate.
        decay_rate (float): the decay rate.
        verbose (bool): whether to print output during training.
        shuffle (bool): whether to shuffle the batches every epoch.

    Returns:
        History: the History object generated after training the model.
    """
    callbacks = []
    if early_stopping and validation_data is not None:
        callbacks.append(K.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=patience))
    if learning_rate_decay and validation_data is not None:
        def schedule(epoch):
            """Inverse time decay of the learning rate."""
            return alpha / (1 + decay_rate * epoch)
        callbacks.append(K.callbacks.LearningRateScheduler(schedule,
                                                           verbose=1))
    history = network.fit(data, labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          callbacks=callbacks,
                          verbose=verbose,
                          shuffle=shuffle)
    return history
