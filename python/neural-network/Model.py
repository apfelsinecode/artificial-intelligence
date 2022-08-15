import numpy as np
from tqdm import trange, tqdm

from MeanSquaredError import MeanSquaredError
from NeuralNetwork import NeuralNetwork
from Optimizers import Optimizer

from typing import Iterable

from time import sleep


class Model:

    def __init__(self,
                 net: NeuralNetwork,
                 loss_func: MeanSquaredError,
                 optimizer: Optimizer
                 ):
        self.net = net
        self.loss_func = loss_func
        self.optimizer = optimizer

    def forward(self, x: np.array) -> np.array:
        return self.net.forward(x)

    def train(self, train_data: Iterable[np.array], train_labels: Iterable[np.array], epochs: int):

        for epoch in range(epochs):
            print("Start epoch", epoch)
            data_and_label = list(zip(train_data, train_labels))
            np.random.shuffle(data_and_label)
            with tqdm(data_and_label) as tz:
                for x, t in tz:
                    pred = self.net.forward(x)
                    loss = self.loss_func.forward(y_pred=pred, y_true=t)
                    tz.set_description(f"loss: {loss}\t")

                    grad = self.loss_func.backward(y_pred=pred, y_true=t)
                    grad, w_grads, b_grads = self.net.backward(x=x, grad=grad)

                    self.optimizer.update(w_grads, b_grads)


    def evaluate_xor(self):
        in_data = [np.array([[0, 0]]),
                   np.array([[0, 1]]),
                   np.array([[1, 0]]),
                   np.array([[1, 1]])]
        labels = [np.array([[0]]),
                  np.array([[1]]),
                  np.array([[1]]),
                  np.array([[0]])]

        for x, t in zip(in_data, labels):
            pred = self.forward(x)
            print(f"x={x}, pred={pred}, true={t}")
