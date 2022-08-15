import numpy as np


class Dropout:

    def __init__(self, p=.5):
        """
        :param p: Probability that a weight is dropped
        """
        self.p = p
        self.mask = None

    def forward(self, x: np.ndarray) -> np.array:
        self.mask = np.random.rand(*x.shape) > self.p  # which to keep
        return x * self.mask / self.p

    def backward(self, grad: np.array = np.array([[1]])) -> np.array:
        return self.mask * grad / self.p
