#!/usr/bin/env python3
"""Module that defines a neural network with one hidden layer."""
import matplotlib.pyplot as plt
import numpy as np


class NeuralNetwork:
    """Defines a neural network with one hidden layer for binary
    classification."""

    def __init__(self, nx, nodes):
        """Initialize a NeuralNetwork.

        Args:
            nx (int): the number of input features.
            nodes (int): the number of nodes in the hidden layer.

        Raises:
            TypeError: if nx or nodes is not an integer.
            ValueError: if nx or nodes is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter for the hidden layer weights."""
        return self.__W1

    @property
    def b1(self):
        """Getter for the hidden layer bias."""
        return self.__b1

    @property
    def A1(self):
        """Getter for the hidden layer activated output."""
        return self.__A1

    @property
    def W2(self):
        """Getter for the output neuron weights."""
        return self.__W2

    @property
    def b2(self):
        """Getter for the output neuron bias."""
        return self.__b2

    @property
    def A2(self):
        """Getter for the output neuron activated output."""
        return self.__A2

    def forward_prop(self, X):
        """Calculate the forward propagation of the neural network.

        Args:
            X (numpy.ndarray): input data of shape (nx, m).

        Returns:
            tuple: the activated outputs __A1 (hidden) and __A2 (output).
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculate the cost of the model using logistic regression.

        Args:
            Y (numpy.ndarray): correct labels of shape (1, m).
            A (numpy.ndarray): activated output of shape (1, m).

        Returns:
            float: the cross-entropy cost.
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """Evaluate the neural network's predictions.

        Args:
            X (numpy.ndarray): input data of shape (nx, m).
            Y (numpy.ndarray): correct labels of shape (1, m).

        Returns:
            tuple: the predicted labels (1 if A2 >= 0.5 else 0) and the cost.
        """
        _, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        prediction = np.where(A2 >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculate one pass of gradient descent on the neural network.

        Args:
            X (numpy.ndarray): input data of shape (nx, m).
            Y (numpy.ndarray): correct labels of shape (1, m).
            A1 (numpy.ndarray): activated output of the hidden layer.
            A2 (numpy.ndarray): predicted output.
            alpha (float): the learning rate.
        """
        m = Y.shape[1]

        dZ2 = A2 - Y
        dW2 = (1 / m) * np.matmul(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.matmul(self.__W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = (1 / m) * np.matmul(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        self.__W2 = self.__W2 - alpha * dW2
        self.__b2 = self.__b2 - alpha * db2
        self.__W1 = self.__W1 - alpha * dW1
        self.__b1 = self.__b1 - alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Train the neural network and optionally print/plot the cost.

        Args:
            X (numpy.ndarray): input data of shape (nx, m).
            Y (numpy.ndarray): correct labels of shape (1, m).
            iterations (int): number of iterations to train over.
            alpha (float): the learning rate.
            verbose (bool): print the cost every step iterations.
            graph (bool): plot the cost every step iterations.
            step (int): interval for printing/plotting.

        Raises:
            TypeError: on wrong types for iterations, alpha or step.
            ValueError: on non-positive values (or step out of range).

        Returns:
            tuple: the evaluation of the training data after training.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        steps = []
        for i in range(iterations + 1):
            A1, A2 = self.forward_prop(X)
            cost = self.cost(Y, A2)
            if (verbose or graph) and (i % step == 0 or i == iterations):
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))
                if graph:
                    costs.append(cost)
                    steps.append(i)
            if i < iterations:
                self.gradient_descent(X, Y, A1, A2, alpha)

        if graph:
            plt.plot(steps, costs, 'b')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)
