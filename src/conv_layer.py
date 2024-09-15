import numpy as np
from scipy import signal
from layer import Layer
import sys
#
# Neural Network Layer for Matrix Multiplication 
#
class ConvLayer(Layer):
    def __init__(self, batch_size, input_shape, kernel_size, num_filters):
        ''''
          batch_size: the number of 
          input_shape: a tuple (channels, height, width)
          num_filters: the number of kernels to be created
          kernel_size: height and width of each kernel
        '''
        self.input_dimensions = len(input_shape)
        if self.input_dimensions == 2:
            self.input_depth = 1
            output_height = input_shape[0] - kernel_size + 1
            output_width = input_shape[1] - kernel_size + 1 
            fan_in = input_shape[0] * input_shape[1] 
        else:
            self.input_depth = input_shape[0]
            output_height = input_shape[1] - kernel_size + 1
            output_width = input_shape[2] - kernel_size + 1  
            fan_in = input_shape[0] * input_shape[1] * input_shape[2] 
            

        self.batch_size = batch_size
        self.number_of_filters = num_filters 
        self.z = np.zeros((batch_size, num_filters, output_height, output_width))

        rng = np.random.default_rng()
        self.weights = rng.normal(loc=0.0, scale=1.0 / ( fan_in** 0.5), size= (num_filters, self.input_depth, kernel_size, kernel_size))
        self.biases = np.zeros(num_filters)

    # Forward pass
    def forward(self, a: np.ndarray, y=None) -> np.ndarray:
        # Compute the feature maps
        for i in range(self.batch_size):
            for j in range(self.number_of_filters):
                if self.input_dimensions == 2:
                    self.z[i, j] = signal.correlate(a[i][np.newaxis,:], self.weights[j], mode="valid", method="auto") + self.biases[j]
                else:
                    self.z[i, j] = signal.correlate(a[i], self.weights[j], mode="valid", method="auto") + self.biases[j]
 
        # save the inputs for the backward pass
        self.a = a 
        
        # return the feature maps
        return self.z
    
    # Backward pass
    def backward(self, djdz: np.ndarray, eta: float) -> np.ndarray:
        djdw = np.zeros(self.weights.shape)
        djda = np.zeros(self.a.shape)
        db = np.zeros(self.biases.shape)
        for i in range(self.batch_size):
            for j in range(self.number_of_filters):
                db[j] += np.sum(djdz[i, j])
                for k in range(self.input_depth):
                    if self.input_dimensions == 2:
                        djdw[j, k] += signal.correlate(self.a[i], djdz[i, j], mode = "valid", method="auto")
                        djda[i] += signal.convolve(djdz[i, j], self.weights[j, k], mode = "full", method="auto")
                    else:
                        djdw[j, k] += signal.correlate(self.a[i,k], djdz[i, j], mode = "valid", method="auto")
                        djda[i, k] += signal.convolve(djdz[i, j], self.weights[j, k], mode = "full", method="auto")

        # Update the weights
        self.weights = self.weights - (eta/self.batch_size) * djdw
        self.biases  = self.biases  - (eta/self.batch_size) * db
        
        return  djda