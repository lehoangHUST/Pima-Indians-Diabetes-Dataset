import numpy as np
from utils.act_functions import get_activation

"""
    Layer : the hidden layer of neural network
    ....

    Attributes
    ----------
        shape: type -> INT
            is the number of neurons in this layer
        activation: type -> STRING
            the activation function of this layer
            default -> 'sigmoid'
"""


class Layer:
    def __init__(self, shape, activation='sigmoid', he_normal='random'):
        self._act_function, self._act_function_der = get_activation(activation)
        self.shape = (shape,)
        self.he_normal = he_normal

    # setup the hidden layer
    # config shape, weights, biases & initialize them
    def _setup(self, prev_layer):
        # Add tuple
        self.shape = (prev_layer.shape[0],) + self.shape
        if self.he_normal == 'random':
            self.weight = np.random.randn(prev_layer.shape[1], self.shape[1])
        elif self.he_normal == 'he':
            self.weight = np.random.randn(prev_layer.shape[1], self.shape[1]) * np.sqrt(2 / prev_layer.shape[1])
        elif self.he_normal == 'xavier':
            self.weight = np.random.randn(prev_layer.shape[1], self.shape[1]) * np.sqrt(1 / prev_layer.shape[1])
        elif self.he_normal == 'zeros':
            self.weight = np.zeros(prev_layer.shape[1], self.shape[1])
        else:
            raise TypeError
        self.bias = np.random.randn(1, self.shape[1])
        self.values = np.zeros(self.shape)

    def _get_spec_number(self, prev_layer):
        return self.shape[1] * prev_layer.shape[1]

    def _foward(self, prev_layer):
        if isinstance(prev_layer, np.ndarray):  # first hidden layer
            self.z = np.dot(prev_layer, self.weight) + self.bias
        else:
            self.z = np.dot(prev_layer.values, self.weight) + self.bias
        self.values = self._act_function(self.z)

    def _backward(self, delta, prev_layer, learning_rate):

        delta = delta * self._act_function_der(self.z)
        # NOT SURE ABOUT THE DERIVATIVE OF BIAS
        # <CHECK-LATER>
        delta_bias = np.sum(delta, axis=0).reshape(1, -1)
        if isinstance(prev_layer, np.ndarray):  # first hidden layer
            weight_der = np.dot(prev_layer.T, delta)
            # print(prev_layer.shape)
        else:
            weight_der = np.dot(prev_layer.values.T, delta)
        self.bias += learning_rate * delta_bias
        delta = np.dot(delta, self.weight.T)
        self.weight += learning_rate * weight_der
        return delta
