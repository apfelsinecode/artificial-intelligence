import numpy as np

from MeanSquaredError import MeanSquaredError
from NeuralNetwork import NeuralNetwork
from Sigmoid import Sigmoid


def main():
    # Network Initialization
    net = NeuralNetwork(2, 1, [2], Sigmoid())
    # Setting the layer weights
    net.layers[0].weights = np.array([[0.5, 0.75], [0.25, 0.25]])
    net.layers[1].weights = np.array([[0.5], [0.5]])
    # Loss
    loss_function = MeanSquaredError()
    # Input
    x = np.array([[1, 1]])
    y = np.array([[0]])
    # Forward Pass
    pred = net.forward(x)
    # Loss Calculation
    loss = loss_function.forward(pred, y)
    print(f"Prediction: {pred}")
    print(f"Loss: {loss}")

    grad = loss_function.backward(pred, y)
    grad, W_grads, b_grads = net.backward(x, grad)
    print(f"Gradients of the first layer: W1: {W_grads[0]}, b1: {b_grads[0]}")
    print(f"Gradients of the second layer: W2: {W_grads[1]}, b2 {b_grads[1]}")


if __name__ == '__main__':
    main()
