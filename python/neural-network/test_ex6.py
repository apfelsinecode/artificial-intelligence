import numpy as np

from MeanSquaredError import MeanSquaredError
from Model import Model
from NeuralNetwork import NeuralNetwork
from Sigmoid import Sigmoid
from Optimizers import SGD, Momentum, Adam


def test_sgd():
    ...


def main():
    # Network Initialization
    net = NeuralNetwork(input_size=2, output_size=1, hidden_size=[2], activation=Sigmoid())
    # Setting the layer weights
    net.layers[0].weights = np.array([[0.5, 0.75], [0.2, 0.3]])
    net.layers[1].weights = np.array([[0.4], [0.6]])
    # Loss
    loss_function = MeanSquaredError()

    # Optimizer
    # optimizer = SGD(nn=net, lr=0.5)
    optimizer = Momentum(nn=net, mu=0.2, lr=0.1)
    # optimizer = Adam(nn=net, lr=0.001, beta1=0.9, beta2=0.999)

    # input
    in_data = [np.array([[0, 0]]),
               np.array([[0, 1]]),
               np.array([[1, 0]]),
               np.array([[1, 1]])] * 10

    labels = [np.array([[0]]),
              np.array([[1]]),
              np.array([[1]]),
              np.array([[0]])] * 10
    print(str(net))
    model = Model(net=net, loss_func=loss_function, optimizer=optimizer)
    model.train(in_data, labels, epochs=1000)
    # for x, t in zip(in_data, labels):
    #     print("\n========\n")
    #     pred = net.forward(x)
    #     loss = loss_function.forward(y_pred=pred, y_true=t)
    #
    #     print(f"input={x}, pred={pred}, target={t}, loss={loss}")
    #     grad = loss_function.backward(y_pred=pred, y_true=t)
    #     grad, w_grads, b_grads = net.backward(x=x, grad=grad)
    #
    #     optimizer.update(w_grads=w_grads, b_grads=b_grads)
    print(str(net))
    model.evaluate_xor()


if __name__ == '__main__':
    main()
