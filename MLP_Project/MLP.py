import math
import random

class MLP():
    def __init__(self, neurons, biasTrue):
        self.num_neurons = neurons
        self.bias = biasTrue

    def forwardPropagation(self, values):
        for i in range(len(self.num_neurons)):
            output_neutron_values = []
            for j in range(self.num_neurons[i]):
                neuron_calculation = 0
                for k in range(len(values)):
                    neuron_calculation = neuron_calculation + (random.uniform(-1, 1) * values[k])
                    if self.bias:
                        neuron_calculation = neuron_calculation + random.uniform(-0.2,0.2)
                neuron_calculation = -neuron_calculation
                output_neutron_values.append(1/(1 + math.exp(neuron_calculation)))
            values = output_neutron_values
        print(values)




