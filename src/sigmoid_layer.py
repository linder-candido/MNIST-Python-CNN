import numpy as np
#
# Neural Network Layer for Matrix Multiplication 
#
class SigmoidLayer:
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
    def forward(self, u, y):
        z = np.dot(self.weights, u) + self.biases  
        a = self.sigmoid(z)
        
        # For the backward pass)
        self.u = u
        self.dsigmoid = a * (1.0 - a) 
 
        return a
    
    # Backward pass
    def backward(self, djda: np.ndarray, eta: float) -> np.ndarray:
        djdz = djda * self.dsigmoid             
        djdu = np.dot(self.weights.T , djdz)
        djdw = np.dot(djdz, self.u.T)

        # Update the weights
        db = np.sum(djdz, axis=1).reshape(self.biases.shape)
        self.weights = self.weights - (eta/self.batch_size) * djdw
        self.biases  = self.biases  - (eta/self.batch_size) * db
        
        return  djdu
    
    # Logistic function
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))