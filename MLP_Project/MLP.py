import random
import ioFunctions
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

    def count_error(self, output):
        error = 0
        for i, neuron in enumerate(self.layers[-1]):
            error += ((neuron.output - output[i])**2)/2

        error = error/len(self.layers[-1])
        return error

    def forwardPropagation(self, values):
        values = np.array(values)
        for layer in self.layers:
            for neuron in layer:
                neuron.addValues(values)
            values = [neuron.output for neuron in layer]
        return np.array(values)

    def derivSigmoid(self, s):
        return s * (1 - s)

    def backPropagation(self, output_value_predicted):
        last_layer_error_signal = np.array([(output_value_predicted[i] - neuron.output) * neuron.derivSigmoid(neuron.output)
                                            for i, neuron in enumerate(self.layers[-1])])
        errors = np.array([last_layer_error_signal[i] * neuron.input_values
                           for i, neuron in enumerate(self.layers[-1])])
        weights_grads = []
        bias_grads = []
        weights_grads.append(errors)
        bias_grads.append(last_layer_error_signal)
        last_grad = errors
        last_layer = self.layers[-1]
        for hid_layer in reversed(self.layers[:-1]):
            last_later_weights = np.array([neuron.weights for neuron in last_layer])
            last_layer_error_signal = np.array([np.dot(last_later_weights[:, i], last_grad[:, i]) * neuron.derivSigmoid(neuron.output)
                                                for i, neuron in enumerate(hid_layer)])
            errors = np.array([last_layer_error_signal[i] * neuron.input_values
                               for i, neuron in enumerate(hid_layer)])
            weights_grads.append(errors)
            bias_grads.append(last_later_weights)
            last_grad = errors
            last_layer = hid_layer

        return weights_grads[::-1], bias_grads[::-1]

    def updating_weights(self, learning_rate, momentum, bias_grads, weights_grads):
        weight = [np.zeros_like(layer[0].weights) for layer in self.layers]
        bias = [np.zeros_like(layer[0].biases) for layer in self.layers]
        for i, layer in enumerate(self.layers):
            for j, neuron in enumerate(layer):
                weight[i][j] = momentum * weight[i][j] - learning_rate * weights_grads[i][j]
                bias[i][j] = momentum * bias[i][j] - learning_rate * bias_grads[i][j]
                neuron.weights += weight[i][j]
                neuron.biases += bias[i][j]

    def train(self, data_set, shuffle=None, stop_value=None, number_of_epochs=None):
        train_error = 0
        epoch = 1
        if shuffle is None:
            random.shuffle(data_set)
        if (stop_value and number_of_epochs) or (number_of_epochs is None and stop_value is None):
            raise ValueError("Error")
        if stop_value:
            while(stop_value > train_error):
                if epoch % 20 == 0:
                    ioFunctions.writeStats("stats", "Epoch: " + str(epoch))
                    train_error = self.epoch(data_set, True)
                else:
                    train_error = self.epoch(data_set)
                epoch += 1
        if number_of_epochs:
            for i in range(number_of_epochs):
                if epoch % 20 == 0:
                    ioFunctions.writeStats("stats","Epoch: " + str(epoch))
                    train_error = self.epoch(data_set, True)
                else:
                    train_error = self.epoch(data_set)
                epoch += 1
        return


    def epoch(self, data_set, stats=None):
        valid_error = 0
        correct_train_predictions = 0
        for i, data in enumerate(data_set):
            result = self.forwardPropagation(data[0])
            if stats is not None:
                if result.argmax() == np.array(data[1]).argmax():
                    correct_train_predictions += 1
            valid_error += self.count_error(data[i])
            weight, bias = self.backPropagation(data[1])
            self.updating_weights(weight, bias)

        if stats is not None:
            ioFunctions.writeStats("stats", "Correct predictions: " + str(correct_train_predictions) + " Epoch error: " + str(valid_error))

        return valid_error





