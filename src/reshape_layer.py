import numpy as np
import sys
from layer import Layer

#
# This layer is used to "flatten" the feature maps reshaping it as a 1d array
#

class ReshapeLayer(Layer):
    def __init__(self, input_shape):
        self.batch_size, self.depth, self.height, self.width = input_shape
        self.size =  self.depth * self.height * self.width

    def forward(self, input: np.ndarray):
        return np.reshape(input, (self.batch_size, self.size)).T

    def backward(self, djda, y=None, eta=None):
        return np.reshape(djda.T, (self.batch_size, self.depth, self.height, self.width))