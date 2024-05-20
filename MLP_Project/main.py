import pickle
import numpy as np
import MLP
from ucimlrepo import fetch_ucirepo
from sklearn.metrics import confusion_matrix

print("Multi layer Perceptron program")
print("1. Iris dataset")
print("2. Autoencoder")
choose = int(input("Enter your choice: "))
if choose == 1:
    iris = fetch_ucirepo(id=53)

    X = iris.data.features
    Y = iris.data.targets

    values = X.to_numpy()
    target = Y.to_numpy()

    target_values = []
    for genre in target:
        if genre == "Iris-setosa":
            target_values.append([1, 0, 0])
        elif genre == "Iris-versicolor":
            target_values.append([0, 1, 0])
        elif genre == "Iris-virginica":
            target_values.append([0, 0, 1])

    target_values = np.array(target_values)

    combined_data = np.concatenate((values, target_values), axis=1)
    training_data = np.concatenate((combined_data[0:20], combined_data[50:70], combined_data[100:120]), axis=0)
    test_data = np.concatenate((combined_data[20:50], combined_data[70:100], combined_data[120:150]), axis=0)
elif choose == 2:
    x_array = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])
    y_array = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])
    combined_data = np.concatenate((x_array, y_array), axis=1)
    training_data = combined_data
    test_data = combined_data

exitValue = False
while exitValue == False:
    print("Choose what you want to do")
    print("1. Create network")
    print("2. Train network")
    print("3. Test network")
    print("4. Save network")
    print("5. Load network")
    print("6. Exit")
    choose1 = int(input("Enter your choice: "))

    if choose1 == 1:
        hidden_layer_neurons = []
        input_layer_size = int(input("Enter the input layer size"))
        hidden_layer_size = int(input("Enter the hidden layer size"))
        for i in range(hidden_layer_size):
            hidden_layer_neurons.append(int(input("Enter the number of neurons on " + i + " hidden layer")))
        output_layer_size = int(input("Enter the output layer size"))
        print("Add bias?")
        print("1. Yes")
        print("2. No")
        choose2 = int(input("Enter your choice: "))
        if choose2 == 1:
            network = MLP.MLP(input_layer_size, hidden_layer_neurons, output_layer_size, True)
        elif choose2 == 2:
            network = MLP.MLP(input_layer_size, hidden_layer_neurons, output_layer_size, False)
    elif choose1 == 2:
        print("Choose stopValue or number of epoch")
        print("1. Number of epoch")
        print("2. Stop Value")
        choose3 = int(input("Enter your choice: "))
        if choose3 == 1:
            epoch = int(input("Enter number of epoch"))
            stopValue = None
        elif choose3 == 2:
            stopValue = int(input("Enter stopValue"))
            epoch = None
        print("Shuffle dataSet?")
        print("1. Yes")
        print("2. No")
        choose4 = int(input("Enter your choice: "))
        if choose4 == 1:
            shuffle = True
        elif choose4 == 2:
            shuffle = None
        network.train(training_data, shuffle, stopValue, epoch)
    elif choose1 == 3:
        rangeLoop = 0
        true_labels = []
        predicted_labels = []
        if choose == 1:
            correct = [0,0,0]
            rangeLoop = 90
        elif choose == 2:
            correct = [0,0,0,0]
            rangeLoop = 4
        for i in range(rangeLoop):
            test = test_data[i]
            predicted = network.forwardPropagation(test[:4])
            if choose == 1:
                expected = test[:-3]
            elif choose == 2:
                expected = test[:-4]
            predicted_label = np.argmax(expected)
            true_label = np.argmax(predicted)
            true_labels.append(true_label)
            predicted_labels.append(predicted_label)
            if predicted_label == true_label:
                correct[true_label] += 1
            error = network.count_error(expected)
            outputs = []
            weights = []
            for i in range(network.layers):
                layer_outputs = []
                layer_weights = []
                for neuron in network.layers[i]:
                    layer_outputs.append(network.layers[i].neuron.output)
                    layer_weights.append(network.layers[i].neuron.weights)
                outputs.append(layer_outputs)
                weights.append(layer_weights)

            with open("trainStats.txt", "a") as file:
                file.write(f"Wzorzec numer: {i}, {test[:4]}\n")
                file.write(f"Popełniony błąd dla wzorca: {error}\n")
                file.write(f"Pożądany wzorzec odpowiedzi: {expected}\n")
                for i in range(len(predicted)):
                    file.write(f"Błąd popełniony na {i} wyjściu: {predicted[i] - expected[i]}\n")
                for i in range(len(predicted)):
                    file.write(f"Wartość na {i} wyjściu: {predicted[i]}\n")
                file.write(f"Wartości wag neuronów wyjściowych\n {weights[-1]}\n")
                for i in reversed(range(len(network.layers) - 1)):
                    file.write(f"Wartości wyjściowe neuronów ukrytych warstwy {i}: {outputs[i]}\n")
                for i in reversed(range(len(network.layers) - 1)):
                    file.write(f"Wartości wag neuronów ukrytych warstwy {i}:\n {weights[i]}\n")
                file.write("\n\n")

        if choose == 1:
            print("Klasyfikacja irysow")
            accuracy = sum(correct) / (len(test_data)) * 100
            print("Iris-setosa: " + str(correct[0] / 35 * 100) + "%")
            print("Iris-versicolor: " + str(correct[1] / 35 * 100) + "%")
            print("Iris-virginica: " + str(correct[2] / 35 * 100) + "%")
            print("Total: " + str(accuracy) + "%")
        else:
            print("Autoenkoder")
            print("Popelniony blad: ", error)
            print("Odpowiedzi: ", predicted)
            print("Poprawne odpowiedzi: ", expected)

        matrix = confusion_matrix(true_labels, predicted_labels)
        print("\nMacierz pomyłek:")
        print(matrix)
        precision = np.diag(matrix) / np.sum(matrix, axis=0)
        recall = np.diag(matrix) / np.sum(matrix, axis=1)
        f_measure = 2 * (precision * recall) / (precision + recall)

        print("\nPrecyzja (Precision):", precision)
        print("Czułość (Recall):", recall)
        print("Miara F (F-measure):", f_measure)

    elif choose1 == 4:
        with open("network.pkl", "wb") as f:
            pickle.dump(network, f)

    elif choose1 == 5:
        with open("network.pkl", "rb") as f:
            network = pickle.load(f)

    elif choose1 == 6:
        exitValue = False



