import random
import numpy as np

class neuron():

    def __init__(self, input_values=None, weights=None, biasTrue=None):
        self.weights = weights
        if len(weights) != len(input_values):
            numberOfWeightsToGenerate = len(input_values) - len(weights)
            if numberOfWeightsToGenerate > 0:
                for i in range(numberOfWeightsToGenerate):
                    self.weights.append(random.uniform(-1, 1))
            else:
                print("Error: numberOfWeightsToGenerate must be a positive integer")
        if biasTrue is not None:
            self.bias = random.uniform(-0.2,0.2)
        self.input_values = input_values
        if input_values is None:
            self.output = 0
        else:
            neuron_calculation = self.add_weights()
            self.output = self.sigmoid(neuron_calculation)


    def add_weights(self):
        return np.dot(self.input_values, self.weights) + self.bias

    def sigmoid(self, calculation):
        return 1 / (1 + np.exp(-calculation))

    def addValues(self, values):
        if values is not None:
            if len(values) == len(self.weights):
                self.input_values = values
                neuron_calculation = self.add_weights()
                self.output = self.sigmoid(neuron_calculation)


