# Problem Statement : Build a forward pass for a 
# 3-4-4-2 network. Feed it RGB color values 
# (normalized to 0-1) and observe the two outputs. 
# This is the architecture for a simple color classifier 
# with two classes.


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


# Create a 3-4-4-2 neural network
hidden_layer1 = Layer(3, 4)
hidden_layer2 = Layer(4, 4)
output_layer = Layer(4, 2)

color_network = Network([
    hidden_layer1,
    hidden_layer2,
    output_layer
])

# RGB color values normalized to the range [0, 1]
rgb_data = [
    ("Red", [1.0, 0.0, 0.0]),
    ("Green", [0.0, 1.0, 0.0]),
    ("Blue", [0.0, 0.0, 1.0]),
    ("Yellow", [1.0, 1.0, 0.0]),
    ("White", [1.0, 1.0, 1.0]),
    ("Black", [0.0, 0.0, 0.0]),
    ("Purple", [0.5, 0.0, 0.5])
]

# Run forward propagation
for color_name, inputs in rgb_data:
    result = color_network.forward(inputs)

    print(f"Color: {color_name}")
    print(f"RGB Input: {inputs}")
    print(f"Hidden Layer 1: {hidden_layer1.last_output}")
    print(f"Hidden Layer 2: {hidden_layer2.last_output}")
    print(f"Final Output: {result}")
    print("-" * 50)
