import numpy as np
import sys
#
# Neural Network Layer for Matrix Multiplication 
#
class SoftmaxLayer:
    def __init__(self, inputs, outputs, batch_size):
        ''''
          inputs:  dimension of the input vector.
          outputs: dimension of the output vector.
        '''
        if (inputs > 0) and (outputs > 0):  
            self.batch_size = batch_size
            self.size = outputs
            rng = np.random.default_rng()
            self.biases = rng.normal(loc=0.0, scale=1.0,size=(outputs, 1))
            self.weights = rng.normal(loc=0.0, scale=1.0/np.sqrt(inputs),size=(outputs, inputs))

    # Forward pass
    def forward(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        z = np.dot(self.weights, u) + self.biases  
        a = self.softmax(z)
        
        # save the inputs for the backward pass
        self.u = u 
        na = self.weights.shape[0] # number of activations
        Sy = a[y, np.arange(a.shape[1])] # softmax for the correct class
       
        # Compute the derivative w.r.t each input in the batch for the backward pass
        self.dsoftmax = Sy * ((np.tile(y, (na, 1)) == np.tile(np.arange(na).reshape(na,1), (1,self.batch_size))) - a)
        return a
    
    # Backward pass
    def backward(self, djda: np.ndarray, eta: float) -> np.ndarray:
        djdz = self.dsoftmax * djda   
        djdu = np.dot(self.weights.T , djdz)
        djdw = np.dot(djdz, self.u.T)

        # Update the weights
        db = np.sum(djdz, axis=1).reshape(self.biases.shape)
        self.weights = self.weights - (eta/self.batch_size) * djdw
        self.biases  = self.biases  - (eta/self.batch_size) * db
        
        return  djdu
    
    # Logistic function
    def softmax(self, z):
        t = np.exp(z - np.max(z))
        return t / np.sum(t, axis=0)