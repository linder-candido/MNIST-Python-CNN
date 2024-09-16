import numpy as np
from layer import Layer
#
# Sigmoid activation implemented as a layer 
#
class SigmoidLayer(Layer):
    # Sigmoid function
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    
    # Forward pass
    def forward(self, z):
        a = self.sigmoid(z)
        self.a = a # save for the backword pass
        return a
    
    # Backward pass
    def backward(self, djda: np.ndarray, y=None, eta=None) -> np.ndarray:
        dsigmoid = self.a * (1.0 - self.a)             
        
        # return to the previous layer
        return  djda * dsigmoid