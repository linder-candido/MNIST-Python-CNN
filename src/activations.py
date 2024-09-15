import numpy as np
import sys
from layer import Layer

class relu(Layer):
    def forward(self, input: np.ndarray, y: np.ndarray=None):
        self.input = input # for the backward pass
        return np.maximum(0, input)

    def backward(self, upstream_gradient: np.ndarray, eta: float = None):
        drelu = np.where(self.input > 0, 1, 0)
        return upstream_gradient * drelu

        
