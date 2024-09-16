import numpy as np
from layer import Layer
#
# Implement a Fully-Conected Layer 
#
class FullyConectedLayer(Layer):
    def __init__(self, inputs, outputs, batch_size):
        ''''
          inputs:  number of neurons in the previous layer.
          outputs: number of neuron of this layer.
        '''
        if (inputs > 0) and (outputs > 0):  
            self.batch_size = batch_size
            self.size = outputs
            rng = np.random.default_rng()
            self.biases = rng.normal(loc=0.0, scale=1.0, size=(outputs, 1))
            self.weights = rng.normal(loc=0.0, scale=1.0/np.sqrt(inputs), size=(outputs, inputs))

    # Forward pass
    def forward(self, a: np.ndarray) -> np.ndarray:
        z = np.dot(self.weights, a) + self.biases  
        self.a = a  # save for the backward pass
        
        # return for the next layer
        return z
    
    # Backward pass
    def backward(self, djdz: np.ndarray, y=None, eta: float=None) -> np.ndarray:            
        djda = np.dot(self.weights.T , djdz)
        djdw = np.dot(djdz, self.a.T)

        # Update the weights
        dbiases = np.sum(djdz, axis=1).reshape(self.biases.shape)
        self.weights = self.weights - (eta/self.batch_size) * djdw
        self.biases  = self.biases  - (eta/self.batch_size) * dbiases
        
        # return for the previous layer
        return  djda