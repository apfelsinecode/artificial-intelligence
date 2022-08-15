import numpy as np

from FullyConnectedLayer import FullyConnectedLayer
from Sigmoid import Sigmoid


class NeuralNetwork:

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_size: list[int],
                 activation=Sigmoid()):

        self.layer_inputs = None
        self.activation_inputs = None
        self.layers = list()
        in_size = input_size
        out_size: int
        for size in [*hidden_size, output_size]:
            self.layers.append(FullyConnectedLayer(in_size, size))
            in_size = size
        self.activation = activation

    def forward(self, x: np.array) -> np.array:
        current = x
        self.layer_inputs = []
        self.activation_inputs = []
        for layer in self.layers[:-1]:
            self.layer_inputs.append(current)
            current = layer.forward(current)
            self.activation_inputs.append(current)
            current = self.activation.forward(current)

        self.layer_inputs.append(current)
        current = self.layers[-1].forward(current)
        return current

    def backward(self, x: np.array, grad: np.array = np.array([[1]])) -> (np.array, list[np.array], list[np.array]):
        weight_grads = []
        bias_grads = []
        # incoming_grad = grad

        incoming_grad, w_grad, b_grad = self.layers[-1].backward(x=self.layer_inputs[-1], grad=grad)
        weight_grads.append(w_grad)
        bias_grads.append(b_grad)

        for layer, layer_input, activation_input \
                in reversed(list(zip(self.layers, self.layer_inputs, self.activation_inputs))):
            # activation_inputs is one shorter than the others
            incoming_grad = self.activation.backward(x=activation_input, grad=incoming_grad)
            incoming_grad, w_grad, b_grad = layer.backward(x=layer_input, grad=incoming_grad)
            weight_grads.append(w_grad)
            bias_grads.append(b_grad)
        weight_grads.reverse()
        bias_grads.reverse()
        return incoming_grad, weight_grads, bias_grads

    def __str__(self):
        result = f"NeuralNetwork: activation={self.activation}, layers:"
        for index, layer in enumerate(self.layers):
            result += f"\nlayer #{index}:\t{str(layer)}"
        return result
