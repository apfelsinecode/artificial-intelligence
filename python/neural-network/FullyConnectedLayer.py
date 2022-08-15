import numpy as np

from Layer import Layer


class FullyConnectedLayer(Layer):

    def __init__(self, input_size: int, output_size: int):
        self._weights = np.zeros(shape=(input_size, output_size))
        self._bias = np.zeros(shape=(1, output_size))

    def forward(self, x: np.array) -> np.array:
        return x @ self._weights + self._bias

    def backward(self, x: np.array, grad: np.array) -> (np.array, np.array, np.array):
        """
        dL/dX = dY/dT * W^T

        dL/dW = x^T * dY/dT

        dL/dbias = dY

        :param x: the input from the prev. layer
        :param grad: the requested change (gradient) from the next layer (dY/dL)
        :return: gradient input, gradient weight matrix, gradient bias
        """
        grad_input = grad @ self._weights.T
        grad_weight = x.T @ grad
        return grad_input, grad_weight, grad

    def __str__(self):
        return f"FullyConnectedLayer:\n" \
               f"weights={self.weights} (shape={self.weights.shape})\n" \
               f"bias={self.bias} (shape={self.bias.shape})"

    @property
    def weights(self) -> np.array:
        return self._weights

    @property
    def bias(self) -> np.array:
        return self._bias

    @weights.setter
    def weights(self, value: np.array):
        self._weights = value

    @bias.setter
    def bias(self, value: np.array):
        self._bias = value
