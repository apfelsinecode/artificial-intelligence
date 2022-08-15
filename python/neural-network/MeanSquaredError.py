import numpy as np


class MeanSquaredError:

    def __init__(self):
        pass

    def forward(self, y_pred: np.array, y_true: np.array) -> np.float64:
        """
        se = 1/2 (t - o)²

        mse = 1/n sum_{i=1}^n se_i
        :param y_pred:
        :param y_true:
        :return: result
        """
        assert len(y_pred) == len(y_true)
        return np.sum(0.5 * np.square((y_true - y_pred))) / len(y_true)

    def backward(self, y_pred: np.array, y_true: np.array, grad: np.array = np.array([[1]])) -> np.array:
        return y_pred - y_true

