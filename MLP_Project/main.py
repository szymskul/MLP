import pickle
import ssl
import numpy as np
from matplotlib import pyplot as plt

import MLP
from ucimlrepo import fetch_ucirepo

ssl._create_default_https_context = ssl._create_unverified_context

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

    combined_data = list(zip(values, target_values))
    training_data = combined_data[0:20] + combined_data[50:70] + combined_data[100:120]
    test_data = combined_data[20:50] + combined_data[70:100] + combined_data[120:150]
elif choose == 2:
    x_array = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])
    y_array = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])
    combined_data = list(zip(x_array, y_array))
    training_data = combined_data
    test_data = combined_data
else:
    raise ValueError("Invalid choice")
exitValue = False
network = None
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
        if choose == 1:
            input_layer_size = 4
            output_layer_size = 3
        elif choose == 2:
            input_layer_size = 4
            output_layer_size = 4
        else:
            raise ValueError("Invalid choice")
        hidden_layer_size = int(input("Enter the hidden layer size: "))
        for i in range(hidden_layer_size):
            hidden_layer_neurons.append(int(input(f"Enter the number of neurons on hidden layer number {i+1} : ")))
        print("Add bias?")
        print("1. Yes")
        print("2. No")
        choose2 = int(input("Enter your choice: "))
        if choose2 == 1:
            bias = True
        elif choose2 == 2:
            bias = False
        else:
            raise ValueError("Invalid choice")
        network = MLP.MLP(input_layer_size, hidden_layer_neurons, output_layer_size, bias)
    elif choose1 == 2:
        print("Choose stopValue or number of epoch")
        print("1. Number of epoch")
        print("2. Stop Value")
        choose3 = int(input("Enter your choice: "))
        if choose3 == 1:
            epoch = int(input("Enter number of epoch: "))
            stopValue = None
        elif choose3 == 2:
            stopValue = float(input("Enter stopValue: "))
            epoch = 0
        else:
            raise ValueError("Invalid choice")
        print("Shuffle dataSet?")
        print("1. Yes")
        print("2. No")
        choose4 = int(input("Enter your choice: "))
        if choose4 == 1:
            shuffle = True
        elif choose4 == 2:
            shuffle = None
        else:
            raise ValueError("Invalid choice")
        learning_rate = float(input("Enter learning rate: "))
        momentum = float(input("Enter momentum: "))
        train_error_list, correct_predictions_list, number_of_epoch = network.train(data_set=training_data, number_of_epochs=epoch, momentum=momentum, learning_Rate=learning_rate, bias=bias, shuffle=shuffle, stop_value=stopValue)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(range(1, number_of_epoch), train_error_list, linestyle='-', color='b')
        plt.title('Błędy treningowe w zależności od liczby epok')
        plt.xlabel('Liczba epok')
        plt.ylabel('Błąd treningowy')
        plt.ylim(0,1)
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(range(1, number_of_epoch), correct_predictions_list, linestyle='-', color='g')
        plt.title('Dokładność w zależności od liczby epok')
        plt.xlabel('Liczba epok')
        plt.ylabel('Dokładność')
        plt.ylim(0, 1)
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    elif choose1 == 3:
        network.test(test_data, "stats2", choose)
    elif choose1 == 4:
        with open("network.pkl", "wb") as f:
            pickle.dump(network, f)
    elif choose1 == 5:
        with open("network.pkl", "rb") as f:
            network = pickle.load(f)
    elif choose1 == 6:
        exitValue = True
    else:
        raise ValueError("Invalid choice")


