import abc
import numpy as np


class Layer(abc.ABC):

    @abc.abstractmethod
    def forward(self, x: np.array) -> np.array:
        ...

    @abc.abstractmethod
    def backward(self, x: np.array, grad: np.array) -> np.array:
        ...

    @property
    @abc.abstractmethod
    def weights(self) -> np.array:
        ...

    @weights.setter
    @abc.abstractmethod
    def weights(self, value: np.array):
        ...

    @property
    @abc.abstractmethod
    def bias(self) -> np.array:
        ...

    @bias.setter
    @abc.abstractmethod
    def bias(self, value: np.array):
        ...
