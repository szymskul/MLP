import random

from sklearn.metrics import confusion_matrix

import ioFunctions
import neuron
import numpy as np

class MLP():
    def __init__(self, input_layer_size, hidden_layers, output_layer_size ,biasTrue):
        self.input_layer_size = input_layer_size
        all_layers = hidden_layers + [output_layer_size]

        last_layer_size = input_layer_size
        self.layers = [np.empty(layer_size, dtype=neuron.neuron) for layer_size in all_layers]
        for layer_index, layer_size in enumerate(all_layers):
            for i in range(layer_size):
                self.layers[layer_index][i] = neuron.neuron(last_layer_size, biasTrue=biasTrue)
            last_layer_size = layer_size

        self.biasTrue = biasTrue

    def count_error(self, exp_output):
        error = 0
        for i, neuron in enumerate(self.layers[-1]):
            error += ((neuron.output - exp_output[i]) ** 2) / 2
        error = error / len(self.layers[-1])
        return error

    def forwardPropagation(self, values_input):
        values = np.array(values_input)
        for layer in self.layers:
            for neuron in layer:
                neuron.activateFunction(values)
            values = [neuron.output for neuron in layer]

        return np.array(values)

    def backPropagation(self, output_value_predicted):
        weight_grads = []
        bias_grads = []
        last_layer = self.layers[-1]
        last_layer_grad = np.array([
            (neuron.output - output_value_predicted[i]) * neuron.derivSigmoid(neuron.output)
            for i, neuron in enumerate(last_layer)
        ])
        result = np.array([
            last_layer_grad[i] * neuron.input_values
            for i, neuron in enumerate(last_layer)
        ])
        grad_above = result
        weight_grads.append(result)
        bias_grads.append(last_layer_grad)
        for layer in reversed(self.layers[:-1]):
            layer_derivative = np.array([neuron.derivSigmoid(neuron.output) for neuron in layer])
            last_layer_weights = np.array([neuron.weights for neuron in last_layer])
            error = np.array([np.dot(grad_above[:, i], last_layer_weights[:, i]) for i, neuron in enumerate(layer)])
            error2 = np.array([error[i] * neuron.derivSigmoid(neuron.output) for i, neuron in enumerate(layer)])
            result = np.array([error2[i] * neuron.input_values for i, neuron in enumerate(layer)])
            weight_grads.append(result)
            bias_grads.append(error2)
            last_layer = layer
            grad_above = result

        return weight_grads[::-1], bias_grads[::-1]

    def updating_weights(self, learning_rate, momentum, bias_grads, weights_grads, momentum_bias_grads,
                         momentum_weights_grads, biasTrue):
        for i, layer in enumerate(self.layers):
            if momentum_weights_grads is None and momentum_bias_grads is None:
                prev_weights = 0
                prev_bias = 0
            else:
                prev_weights = np.array(momentum_weights_grads[i])
                prev_bias = np.array(momentum_bias_grads[i])

            weight_update = momentum * learning_rate * prev_weights + learning_rate * weights_grads[i]
            if biasTrue is True:
                bias_update = momentum * learning_rate * prev_bias + learning_rate * bias_grads[i]
            for i, neuron in enumerate(layer):
                if biasTrue is True:
                    neuron.bias -= bias_update[i]
                neuron.weights -= weight_update[i]


    def train(self, data_set, number_of_epochs, momentum,  learning_Rate, bias, shuffle=None, stop_value=None):
        train_error = 20
        epoch = 1
        correct_predictions_list = []
        train_error_list = []
        prev_weight = None
        prev_bias = None
        if shuffle is True:
            random.shuffle(data_set)
        if (stop_value and number_of_epochs) or (number_of_epochs is None and stop_value is None):
            raise ValueError("Error")
        if stop_value:
            while(stop_value < train_error):
                if epoch % 20 == 0:
                    ioFunctions.writeStats("stats", "Epoch: " + str(epoch))
                    weight_grad, bias_grad, train_error, correct_predictions = self.epoch(data_set, momentum, learning_Rate, prev_weight, prev_bias, bias, True)
                else:
                    weight_grad, bias_grad, train_error, correct_predictions = self.epoch(data_set, momentum, learning_Rate, prev_weight, prev_bias, bias)
                epoch += 1
                prev_weight = weight_grad
                prev_bias = bias_grad
                train_error_list.append(train_error)
                correct_predictions_list.append(correct_predictions)
        if number_of_epochs:
            for i in range(number_of_epochs):
                if epoch % 20 == 0:
                    ioFunctions.writeStats("stats","Epoch: " + str(epoch))
                    weight_grad, bias_grad, train_error, correct_predictions = self.epoch(data_set, momentum, learning_Rate, prev_weight, prev_bias, bias, True)
                else:
                    weight_grad, bias_grad, train_error, correct_predictions = self.epoch(data_set, momentum, learning_Rate, prev_weight, prev_bias, bias)
                epoch += 1
                prev_weight = weight_grad
                prev_bias = bias_grad
                train_error_list.append(train_error)
                correct_predictions_list.append(correct_predictions)

        return train_error_list, correct_predictions_list, epoch

    def epoch(self, data_set, momentum, learning_rate, prev_weight, prev_bias, bias, stats=None):
        valid_error = 0
        correct_train_predictions = 0
        for i, data in enumerate(data_set):
            result = self.forwardPropagation(data[0])
            if result.argmax() == np.array(data[1]).argmax():
                correct_train_predictions += 1
            weight_grad, bias_grad = self.backPropagation(data[1])
            valid_error += self.count_error(data[1])
            self.updating_weights(learning_rate, momentum, bias_grad, weight_grad, prev_bias, prev_weight, bias)
            prev_weight, prev_bias = weight_grad, bias_grad

        if stats is not None:
            ioFunctions.writeStats("stats", "Correct predictions: " + str(correct_train_predictions) + " Epoch error: " + str(valid_error))

        return prev_weight, prev_bias, valid_error, correct_train_predictions/4


    def test(self, test_set, fileName, which_data_set):
        correct_test_predictions = 0
        predict_outputs = []
        real_outputs = []
        if which_data_set == 1:
            correct = [0,0,0]
        elif which_data_set == 2:
            correct = [0,0,0,0]
        else:
            raise ValueError("Invalid choice")
        for i, data in enumerate(test_set):
            count = self.forwardPropagation(data[0])
            if count.argmax() == np.array(data[1]).argmax():
                correct_test_predictions += 1
            predict_output = np.argmax(count)
            real_output = np.argmax(data[1])
            predict_outputs.append(predict_output)
            real_outputs.append(real_output)
            if predict_output == real_output:
                correct[real_output] += 1
            error = self.count_error(data[1])
            ioFunctions.writeStats(fileName, f"Wzorzec numer {i}")
            ioFunctions.writeStats(fileName, f"Wzorzec wejsciowy {data[0]}")
            ioFunctions.writeStats(fileName, f"Pożądany wzorzec {data[1]}")
            ioFunctions.writeStats(fileName, f"Bład wzorca {error}")
            weights = []
            weights_layer = []
            inputs = []
            inputs_layer = []
            for layer in self.layers:
                for i, neuron in enumerate(layer):
                      weights_layer.append(neuron.weights)
                      inputs_layer.append(neuron.output)
                weights.append(weights_layer)
                inputs.append(inputs_layer)
            ioFunctions.writeStats(fileName, f" ")
            last_layer = self.layers[-1]
            for i, neuron in enumerate(last_layer):
                ioFunctions.writeStats(fileName, f"Błąd na wyjsciu {i} = {count[i] - data[1][i]}")
                ioFunctions.writeStats(fileName, f"Wartosc na wyjsciu {i} = {count[i]}")
            for i in reversed(range(len(self.layers) - 1)):
                ioFunctions.writeStats(fileName, f"Warstwa ukryta numer {i + 1}")
                ioFunctions.writeStats(fileName, f"Wartosci wyjsciowe:  {inputs[i]}")
                ioFunctions.writeStats(fileName, f"Wartosci wag:  {weights[i]}")
            ioFunctions.writeStats(fileName, f"\n")

        if which_data_set == 1:
            print("Irysy")
            print("Całkowity procent poprawnie rozpoznanego gatunku " + str(sum(correct) / (len(test_set)) * 100) + "%")
            print("Iris-setosa: " + str(correct[0] / 30 * 100) + "%")
            print("Iris-versicolor: " + str(correct[1] / 30 * 100) + "%")
            print("Iris-virginica: " + str(correct[2] / 30 * 100) + "%")
        else:
            print("Autoenkoder")
        matrix = confusion_matrix(real_outputs, predict_outputs)
        print("\nMacierz pomyłek:")
        print(matrix)
        precision = np.diag(matrix) / np.sum(matrix, axis=0)
        recall = np.diag(matrix) / np.sum(matrix, axis=1)
        f_measure = 2 * (precision * recall) / (precision + recall)

        print("\nPrecyzja (Precision):", precision)
        print("Czułość (Recall):", recall)
        print("Miara F (F-measure):", f_measure)

