import numpy as np
import pandas as pd

file_data = pd.read_csv('mnist_train.csv')
file_data.div(255)
data = np.array(file_data)

rows, cols = data.shape

np.random.shuffle(data)


''' 
        class Neural_Network

        class attributes:
            data: list of list
            layers: list

        class methods:
            run_batch()
            activation_function()
            generate_weights()
            generate_biases()


'''
class Neural_Network:

    def __init__(self, data, layer_sizes, num_epochs):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.data = data
        self.weights = [] # [15 x 784], [10 x15]
        self.biases = [] 
        self.learing_rate = 3.0
        self.num_epochs = num_epochs


    def activation_function(self, z):
        a = 1 / (1 + np.exp(-z))
        return a

    def run_batch():
        pass

    def generate_weights_and_biases(self):
        for i in range(1, self.num_layers):
            prev_size = self.layer_sizes[i-1]
            #print(prev_size)
            curr_size = self.layer_sizes[i]
            weights = np.random.uniform(-1,1, (curr_size, prev_size))
            biases = np.random.uniform(-1,1, (curr_size,1))
            #print(f"rows = {len(weights)}  Cols = {len(weights[0])}")
            self.weights.append(weights)
            self.biases.append(biases)


n1 = Neural_Network(data, [784,15,10], 30)
n1.generate_weights_and_biases()