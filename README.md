# Simple CNN for MNIST Dataset

This repository contains a simple Convolutional Neural Network (CNN) model implemented from scratch for educational purposes. 
The implementation is done in Python using `numpy` and `scipy` for numerical computations.

## Model Architecture

In order to illustrate fundamental concepts, the CNN model implemented in this repository follows a straightforward architecture 
composed of the following layers: 

1. **Convolutional Layer**:
   - Applies 5 filters (3x3) to extract features from the input images.

2. **ReLU Activation**:
   - Uses the Rectified Linear Unit (ReLU) activation function to introduce non-linearity.

3. **Reshape**:
   - Reshapes the output from the convolutional layer to prepare it for the subsequent fully connected layer.

4. **Fully Connected Layer with Sigmoid Activation**:
   - Applies a fully connected layer followed by a sigmoid activation function to introduce further non-linearity.
   - The ReLu activation is usually the default choice (try it out!). 

5. **Fully Connected Layer**:
   - Another fully connected layer that maps the features to the final classification space.

6. **Softmax Layer**:
   - Applies the softmax activation function to convert the logits into a probability distribution over the digit classes (0-9).

## Purpose

This project is designed to help learners understand the inner workings of a CNN by constructing the model from scratch. 
By implementing each component manually, you will gain insights into the following:

- How convolutional layers extract features from images.
- The role of activation functions in introducing non-linearity.
- How fully connected layers contribute to classification.
- How softmax is used for producing class probabilities.
- How backpropagation is implemented in different types of layers during the backward pass. 

## Technologies Used

- **Python**: The programming language used for the implementation.
- **numpy**: For numerical computations and array manipulations.
- **scipy**: To implement cross-correlation and convolution in the convolutional layer.

## Getting Started

To run the code, you'll need the following Python packages:
- `numpy`
- `scipy`

Install these dependencies using pip:

- pip install numpy scipy 
