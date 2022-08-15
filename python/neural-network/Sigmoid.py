import numpy as np

from Layer import Layer


class Sigmoid(Layer):

    def __init__(self):
        pass

    def forward(self, x: np.array) -> np.array:
        """
        sig(t) = 1 / (1 + e^-t)
        :param x: input
        :return: output
        """
        return self.sigmoid(x)

    def backward(self, x: np.array, grad: np.array = np.array([[1]])) -> np.array:
        """
        dL/dx = [sigmoid(x) * (1 - sigmoid(x))] (*) dL/dy

        dL/dy = grad
        """
        sig_x = self.sigmoid(x)
        return (sig_x * (1 - sig_x)) * grad

    @staticmethod
    def sigmoid(x):
        return np.divide(
            1,
            np.add(1, np.exp(np.negative(x))),
        )

    def weights(self) -> np.array:
        return np.array([])

    def bias(self) -> np.array:
        return []