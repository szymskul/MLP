import MLP
from ucimlrepo import fetch_ucirepo

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

