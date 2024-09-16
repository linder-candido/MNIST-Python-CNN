import numpy as np
from layer import Layer
from fully_conected_layer import FullyConectedLayer
from conv_layer import ConvLayer
from reshape_layer import ReshapeLayer
from relu_layer import ReluLayer
from softmax_layer import SoftmaxLayer
from sigmoid_layer import SigmoidLayer
from mnist_loader  import MnistDataloader

class Network2:
    # Constructor
    def __init__(self) -> None:
        '''' The number of layers does not include the input layer'''
        self.layers = []
        self.number_of_layers = 0

    # Add a layer to the network
    def addLayer(self, layer: Layer):
        self.layers.append(layer)
        self.number_of_layers += 1
    
    
    # Forward pass of a mini-batch throught the network
    def feedForward(self, a: np.ndarray) -> np.ndarray:
        for layer in self.layers:
           a = layer.forward(a)
        return a

    # Backward pass throught the network
    def backprop(self, djda: np.ndarray, y: np.ndarray, eta: float) -> None:
        for layer in reversed(self.layers):
            djda = layer.backward(djda, y, eta)
        
    # Train the network
    def train(self, x_train, labels, epochs, batch_size, eta, test_data = None):
        n = x_train.shape[0]  #x_train shape = (#images, heigh, width)
        for epoch in range(epochs):
            # Shuffle the train set
            perm = np.random.permutation(n)
            x_train = x_train[perm, :, :]
            labels = labels[perm]
            
            # Build the mini-batches 
            batches = [(x_train[k:k+batch_size,:,:], labels[k:k+batch_size]) for k in range(0, n, batch_size)]
            
            i = 0
            for mini_batch in batches:
                i += 1 
                X = mini_batch[0]
                y = mini_batch[1]
                
                # Forward pass through the network
                output = self.feedForward(X)

                # Compute the gradient of the cross-entropy loss
                softmax_y = output[y, np.arange(output.shape[1])] 
                djds = -1.0 / softmax_y
            
                # Backward pass through the network
                self.backprop(djds, y, eta)
                
                if (i % 1000) == 0:
                    print(f'epoca {epoch}: {i} batches processados.')
            
            if test_data is not None:
                print(f'Epoch {epoch} completed: {self.evaluate(test_data[0], test_data[1], batch_size)}, {test_data[1].size}')
            else:
                print(f'Epoch {epoch} completed')
    
        
    # Evaluate the test example
    def evaluate(self, X, y, batch_size): 
        # Builds mini-bathes samples 
        batches = [(X[k:k+batch_size,:,:], y[k:k+batch_size]) for k in range(0, 10000, batch_size)]
            
        # Feed forward the mini batches
        sum = 0 
        for mini_batch in batches: 
            Xb = mini_batch[0]
            yb = mini_batch[1]
            
            # Forward pass
            prob = self.feedForward(Xb)
            y_pred = np.argmax(prob, axis=0)
            sum += np.sum(y_pred == yb)
        
        return sum
                             
# Main function
def main():
    # Create a model
    model = Network2()
    model.addLayer(ConvLayer(4, (28, 28), 3, 5))
    model.addLayer(ReluLayer())
    model.addLayer(ReshapeLayer((4, 5, 26, 26)))
    model.addLayer(FullyConectedLayer(3380, 100, 4))
    model.addLayer(SigmoidLayer())
    model.addLayer(FullyConectedLayer(100, 10, 4))
    model.addLayer(SoftmaxLayer())
     
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