import abc
import math

import numpy as np

from NeuralNetwork import NeuralNetwork


class Optimizer(abc.ABC):

    @abc.abstractmethod
    def update(self, w_grads: list[np.array], b_grads: list[np.array]):
        pass


class SGD(Optimizer):
    """
    Vanilla Gradient Descent: x += -learningrate * df/dx (x)
    """
    def __init__(self, nn: NeuralNetwork, lr: float):
        self.nn = nn
        self.lr = lr

    def update(self, w_grads: list[np.array], b_grads: list[np.array]) -> None:
        assert len(self.nn.layers) == len(w_grads) == len(b_grads)
        for layer, w_grad, b_grad in zip(self.nn.layers, w_grads, b_grads):
            layer.weights -= self.lr * w_grad
            layer.bias -= self.lr * b_grad


class Momentum(Optimizer):
    """
    v = mu * v - learningrate * d/dx f(x)
    mu * v : momentum
    -learningrate * d/dx f(x) : SGD
    """

    def __init__(self, nn: NeuralNetwork, lr: float, mu: float):
        self.nn = nn
        self.lr = lr
        self.mu = mu
        self.v_w = [0] * len(self.nn.layers)
        self.v_b = [0] * len(self.nn.layers)

    def update(self, w_grads: list[np.array], b_grads: list[np.array]) -> None:
        assert len(self.nn.layers) == len(w_grads) == len(b_grads)
        for i in range(len(self.nn.layers)):
            self.v_w[i] = self.mu * self.v_w[i] - self.lr * w_grads[i]
            self.v_b[i] = self.mu * self.v_b[i] - self.lr * b_grads[i]

            self.nn.layers[i].weights += self.v_w[i]
            self.nn.layers[i].bias += self.v_b[i]


class Adam(Optimizer):
    """
    m = beta1 * m + (1 - beta1) * d/dx f(x)
    v = beta2 * v + (1 - beta2) * d/dx f(x)²
    """

    def __init__(self, nn: NeuralNetwork, lr: float, beta1: float, beta2: float):
        self.nn = nn
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.m_w = [0] * len(self.nn.layers)
        self.m_b = [0] * len(self.nn.layers)
        self.v_w = [0] * len(self.nn.layers)
        self.v_b = [0] * len(self.nn.layers)

    def update(self, w_grads: list[np.array], b_grads: list[np.array]) -> None:
        assert len(self.nn.layers) == len(w_grads) == len(b_grads)

        for i in range(len(self.nn.layers)):
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * w_grads[i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * b_grads[i]

            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (w_grads[i] ** 2)
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (b_grads[i] ** 2)

            epsilon = 1e-8

            self.nn.layers[i].weights -= (self.lr / (np.sqrt(self.v_w[i]) + epsilon)) * self.m_w[i]
            self.nn.layers[i].bias -= (self.lr / (np.sqrt(self.v_b[i]) + epsilon)) * self.m_b[i]
