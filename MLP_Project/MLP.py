import neuron
import numpy as np

class MLP():
    def __init__(self, input_layer_size, hidden_layers, output_layer_size ,biasTrue):
        self.input_layer_size = input_layer_size
        next_layer_size = input_layer_size
        self.layers = hidden_layers + output_layer_size
        self.bias = biasTrue
        self.layers = [np.empty(layer_size, dtype=neuron) for layer_size in self.layers]
        for layer in self.layers:
            neurons = len(layer)
            for i in range(neurons):
                layer[i] = neuron.neuron(next_layer_size, None, True)
            next_layer_size = neurons

    def forwardPropagation(self, values):
        values = np.array(values)
        for layer in self.layers:
            for neuron in layer:
                neuron.addValues(values)
            values = [neuron.output for neuron in layer]
        return np.array(values)





