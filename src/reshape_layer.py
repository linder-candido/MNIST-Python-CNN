import numpy as np
import sys
from layer import Layer

class ReshapeLayer(Layer):
    def __init__(self, input_shape):
        self.batch_size, self.depth, self.height, self.width = input_shape
        self.size =  self.depth * self.height * self.width

    def forward(self, input: np.ndarray, y: np.ndarray=None):
        return np.reshape(input, (self.batch_size, self.size)).T

    def backward(self, output_gradient, eta: float=None):
        return np.reshape(output_gradient.T, (self.batch_size, self.depth, self.height, self.width) )