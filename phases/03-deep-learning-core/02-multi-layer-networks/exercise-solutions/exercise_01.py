# Problem Statement : Build a 2-4-2-1 network (two hidden layers) 
# and run the forward pass on XOR data with random weights. 
# Print the intermediate hidden layer outputs to see how the 
# representation transforms at each layer.

import math

def sigmoid(x):
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))

class Layer:
    def __init__(self, n_inputs, n_neurons, weights=None, biases=None):
        if weights is not None:
            self.weights = weights
        else:
            import random
            self.weights = [
                [random.uniform(-1, 1) for _ in range(n_inputs)]
                for _ in range(n_neurons)
            ]
        if biases is not None:
            self.biases = biases
        else:
            self.biases = [0.0] * n_neurons

    def forward(self, inputs):
        self.last_input = inputs
        self.last_output = []
        for neuron_idx in range(len(self.weights)):
            z = sum(
                w * x for w, x in zip(self.weights[neuron_idx], inputs)
            )
            z += self.biases[neuron_idx]
            self.last_output.append(sigmoid(z))
        return self.last_output

class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        return current


hidden_layer1 = Layer(2, 4)
hidden_layer2 = Layer(4, 2)
output_layer = Layer(2, 1)

network = Network([hidden_layer1, hidden_layer2, output_layer])

xor_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0)
]

for inputs, expected in xor_data:
    output = network.forward(inputs)
    print("Hidden Layer 1 Output:", hidden_layer1.last_output)
    print("Hidden Layer 2 Output:", hidden_layer2.last_output)
    print("Final Layer Output:", output_layer.last_output)
    print()