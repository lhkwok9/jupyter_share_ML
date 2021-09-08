import numpy as np
import math
import nnfs

inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

bias = [2.0, 3.0, 0.5]

weights2 = [[0.1, -0.14, 0.5],
           [-0.5, 0.12, -0.33],
           [-0.44, 0.73, -0.13]]

bias2 = [-1, 2, -0.5]

l1_output = np.dot(inputs, np.array(weights).T) + bias
l2_output = np.dot(l1_output, np.array(weights2).T) + bias2

print("l2_output:")
print(l2_output)

x = [[1.0, 2.0, 3.0, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

np.random.seed(0)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_Relu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.forward(x)
layer2.forward(layer1.output)
print("layer2 output:")
print(layer2.output)

nnfs.init()

layer_outputs = [[4.8, 1.21, 2.385],
                [8.9, -1.81, 0.2],
                [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs)
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
print("norm_values:")
print(norm_values)
