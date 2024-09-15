#
# Read MNIST Dataset
#
import numpy as np
import struct
from array import array

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self):
        self.magic1 = 2049
        self.magic2 = 2051
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        
        # Read the labels to a numpy array of dimension (60000,)
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != self.magic1:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = np.array(array("B", file.read()))
        
        # Read the images to a numpy array of dimensions (784, 60000), on image per column
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != self.magic2:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = np.zeros((rows * cols, size))
        for i in range(size):
            images[:,i] = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])          
        
        return (images, labels)
            

#
# Verify Reading Dataset via MnistDataloader class
#
import random
import matplotlib.pyplot as plt
if __name__ == "__main__":
    #
    # Set file paths based on added MNIST Datasets
    #
    training_images_filepath = '../data/train-images.idx3-ubyte'
    training_labels_filepath = '../data/train-labels.idx1-ubyte'
    test_images_filepath = '../data/t10k-images.idx3-ubyte'
    test_labels_filepath = '../data/train-labels.idx1-ubyte'
    
    #
    # Load MINST dataset
    #
    mnist = MnistDataloader()
    (x_train, y_train) = mnist.read_images_labels(training_images_filepath, training_labels_filepath)
    (x_test, y_test) =  mnist.read_images_labels(test_images_filepath, test_labels_filepath)

    #
    # Show some random training and test images 
    #
    images_2_show = []
    titles_2_show = []
    for i in range(10):
        r = random.randint(1, 60000)
        images_2_show.append(x_train[:,r].reshape(28, 28))
        titles_2_show.append('training img [' + str(r) + '] = ' + str(y_train[r]))    

    for i in range(5):
        r = random.randint(1, 10000)
        images_2_show.append(x_test[:,r].reshape(28, 28))        
        titles_2_show.append('test img [' + str(r) + '] = ' + str(y_test[r]))    

    # Show the selected images
    fig, axs = plt.subplots(3, 5)
    index = 0
    for img, txt in zip(images_2_show, titles_2_show):        
        axs[index//5,index%5].set_title(txt)
        axs[index//5,index%5].set_axis_off()        
        axs[index//5,index%5].imshow(img, cmap=plt.cm.gray)
        index += 1
    plt.show()