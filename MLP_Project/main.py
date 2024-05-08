import MLP

mlp = MLP.MLP(neurons=[5, 10, 3], biasTrue=True)
mlp.forwardPropagation(input_values=[0.2,0.4,0.6])
