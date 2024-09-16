import sys
import numpy as np
from layer import Layer
#
# Implements a Softmax layer that works with cross-entropy loss. This
# means the backward pass must receive the gradient of the cross-entropy loss
# w.r.t softmax probabilities as input.   
#
class SoftmaxLayer(Layer):
    # Softmax function
    def softmax(self, z):
        t = np.exp(z - np.max(z))
        return t / np.sum(t, axis=0)

    # Forward pass
    def forward(self, z: np.ndarray) -> np.ndarray:
        a = self.softmax(z)
        self.sftm = a  # for the backward pass

        return a
    
    # Backward pass (softmax coupled with cross-entropy loss)
    def backward(self, djds: np.ndarray, y: np.ndarray, eta=None) -> np.ndarray:
        rows, cols = self.sftm.shape     
        
        # One-Hot-encoded vectors for the true class
        delta = np.zeros((rows, cols))
        delta[y, np.arange(cols)] = 1

        # Softmax for the correct class
        sy = self.sftm[y, np.arange(cols)] 
          
        # Softmax derivative w.r.t inputs
        dsoftmax = sy * (delta - self.sftm)
       
        # Loss derivative w.r.t inputs
        djdz = djds * dsoftmax
        return  djdz
     