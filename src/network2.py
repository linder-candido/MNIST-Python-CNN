import numpy as np
from conv_layer import ConvLayer
from reshape_layer import ReshapeLayer
from activations import relu
from softmax_layer import SoftmaxLayer
from sigmoid_layer import SigmoidLayer
from mnist_loader  import MnistDataloader
import sys

class Network2:
    # Constructor
    def __init__(self) -> None:
        '''' The number of layers does not include the input layer'''
        self.layers = []
        self.number_of_layers = 0

    # Add a convolutional layer to the network
    def addConvLayer(self, batch_size, input_shape, kernel_size, num_filters) -> None:
        self.layers.append(ConvLayer(batch_size, input_shape, kernel_size, num_filters))
        self.number_of_layers += 1
    
    # Add a relu layer to the network
    def addReluLayer(self):
        self.layers.append(relu())
        self.number_of_layers += 1

    def addReshapeLayer(self, input_shape):
        self.layers.append(ReshapeLayer(input_shape))
        self.number_of_layers += 1  

    # Add a fully-connected softmax layer to the network
    def addSigmoidLayer(self, inputs: int, outputs: int, batch_size: int) -> None:
        if (inputs > 0) and (outputs > 0):  
            # Check for compatibility with previous layer
            if self.number_of_layers >= 1:
                if inputs != self.layers[-1].size:
                    raise ValueError("inputs is incompatible with previous layer")
            
            # Compatibility Ok!
            self.layers.append(SigmoidLayer(inputs=inputs, outputs=outputs, batch_size= batch_size))
            self.number_of_layers += 1
        else:
            raise ValueError("inputs and outputs must greater than zero")
    
    
    # Add a fully-connected softmax layer to the network
    def addSoftmaxLayer(self, inputs: int, outputs: int, batch_size: int) -> None:
        if (inputs > 0) and (outputs > 0):  
            # Check for compatibility with previous layer
            if self.number_of_layers >= 1:
                if inputs != self.layers[-1].size:
                    raise ValueError("inputs is incompatible with previous layer")
            
            # Compatibility Ok!
            self.layers.append(SoftmaxLayer(inputs=inputs, outputs=outputs, batch_size= batch_size))
            self.number_of_layers += 1
        else:
            raise ValueError("inputs and outputs must greater than zero")
    
    # Forward pass of a mini-batch throught the network
    def feedForward(self, a: np.ndarray, y: np.ndarray) -> np.ndarray:
        for layer in self.layers:
           a = layer.forward(a, y)
        return a

    # Backward pass of the "error" throught the network
    def backprop(self, djda: np.ndarray, batch_size: int, eta: float) -> None:
        # Update the weights
        for layer in reversed(self.layers):
            djda = layer.backward(djda, eta)
        
    # Train the network
    def train(self, x_train, labels, epochs, batch_size, eta, test_data = None):
        n = x_train.shape[0] 
        for epoch in range(epochs):
            # Shuffles the train set
            perm = np.random.permutation(n)
            x_train = x_train[perm, :, :]
            labels = labels[perm]
            
            # Builds mini-bathes samples 
            batches = [(x_train[k:k+batch_size,:,:], labels[k:k+batch_size]) for k in range(0, n, batch_size)]
            
            # Updates weights using backpropagation
            i = 0
            for mini_batch in batches:
                i += 1 
                X = mini_batch[0]
                y = mini_batch[1]
                
                # Forward pass
                activations = self.feedForward(X, y)

                # Compute gradient of the cross-entropy loss
                Sy = activations[y, np.arange(activations.shape[1])] # softmax for the correct class
                djda = -1.0 / Sy
            
                
                # Update weights using backpropagation
                self.backprop(djda, batch_size, eta)
                if (i % 1000) == 0:
                    print(f'epoca {epoch}: {i} batches processados.')
            
            if test_data is not None:
                print(f'Epoch {epoch} completed: {self.evaluate(test_data[0], test_data[1])}, {test_data[1].size}')
            else:
                print(f'Epoch {epoch} completed')
    
        
    # Evaluate the test example
    def evaluate(self, X, y): 
        # Builds mini-bathes samples 
        batches = [(X[k:k+4,:,:], y[k:k+4]) for k in range(0, 10000, 4)]
            
        # Feed forward the mini batches
        sum = 0 
        for mini_batch in batches: 
            Xb = mini_batch[0]
            yb = mini_batch[1]
            
            # Forward pass
            prob = self.feedForward(Xb, yb)
            y_pred = np.argmax(prob, axis=0)
            sum += np.sum(y_pred == yb)
        
        return sum
                             
# Main function
def main():
    # Create a model
    model = Network2()
    model.addConvLayer(4, (28, 28), 3, 5)
    model.addReluLayer()
    model.addReshapeLayer((4, 5, 26, 26))
    model.addSigmoidLayer(3380, 100, 4)
    model.addSoftmaxLayer(100, 10, 4)
     
    # Read the data set
    training_images_filepath = '../data/train-images.idx3-ubyte'
    training_labels_filepath = '../data/train-labels.idx1-ubyte'
    test_images_filepath = '../data/t10k-images.idx3-ubyte'
    test_labels_filepath = '../data/t10k-labels.idx1-ubyte'
    
    mnist = MnistDataloader()
    (x_train, y_train) = mnist.read_images_labels(training_images_filepath, training_labels_filepath)
    (x_test, y_test) = mnist.read_images_labels(test_images_filepath, test_labels_filepath)

    # Train the model
    x_train /= 255.0
    x_test /= 255.0

    model.train(x_train.T.reshape(60000, 28, 28), y_train, epochs=30, batch_size=4, eta=0.1, test_data=(x_test.T.reshape(10000, 28, 28), y_test))

if __name__ == "__main__":
    main()

            


        
    