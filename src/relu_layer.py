import numpy as np
import sys
from layer import Layer

#
# ReLu activation implemented as layer
#
class ReluLayer(Layer):
    # ReLu function
    def relu(self, z: np.ndarray) -> np.ndarray :
        return np.maximum(0, z) 
    
    # Forward pass
    def forward(self, z: np.ndarray) -> np.ndarray:
        self.z = z   # save the input for the backword pass
        return self.relu(z)
    
    # Backward pass
    def backward(self, djda: np.ndarray, y=None, eta=None) -> np.ndarray:
        # ReLu derivative
        drelu = np.where(self.z > 0, 1, 0)
        
        # return for the previous layer
        return djda * drelu