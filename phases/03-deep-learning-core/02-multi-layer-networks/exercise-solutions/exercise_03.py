# Problem Statement : Implement a count_parameters method 
# on the Network class that returns the total number of 
# trainable weights and biases. Test it on a 784-256-128-10 
# network (the classic MNIST architecture). How many 
# parameters does it have?


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

    def count_parameters(self):
        total = 0
        for layer in self.layers:
            total += sum(len(neuron_weights) 
                for neuron_weights in layer.weights)
            total += len(layer.biases)
        return total

classic_mnist_network = Network([
    Layer(784, 256),
    Layer(256, 128),
    Layer(128, 10)
])

print(classic_mnist_network.count_parameters())

# The total number of parameters is calculated as follows:
# Layer 1: 784 * 256 weights + 256 biases = 200960
# Layer 2: 256 * 128 weights + 128 biases = 32896
# Layer 3: 128 * 10 weights + 10 biases = 1290
# Total = 200960 + 32896 + 1290 = 235146