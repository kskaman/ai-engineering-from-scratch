# Problem Statement : Change the hidden layer size 
# in the circle classifier from 8 to 2, then to 32. 
# Run the forward pass with random weights each time. 
# Does the number of hidden neurons change the output
# range or distribution? Why?


# Increasing the number of hidden neurons does not 
# change the output range because the sigmoid function 
# always gives values between 0 and 1. However, it can 
# change how the output values are distributed because 
# more neurons contribute to the final output.
#
# More hidden neurons allow the network to learn more 
# features and understand complex patterns, which can 
# improve accuracy. However, having more neurons does 
# not always mean better accuracy, especially when the 
# weights are randomly initialized and the network has 
# not been trained.
#
# In conclusion, increasing the number of hidden neurons 
# can help a network learn more complex patterns, but it 
# can also increase the risk of overfitting during training. 
# The network needs proper training and weight adjustments 
# to make accurate predictions on new data.

import math
import random

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



random.seed(42)

data = []
for _ in range(200):
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    label = 1 if (x * x + y * y) < 0.25 else 0
    data.append(([x, y], label))

# Create networks with different hidden layer sizes
circle_net_1 = Network([
    Layer(n_inputs=2, n_neurons=2),
    Layer(n_inputs=2, n_neurons=1),
])

circle_net_2 = Network([
    Layer(n_inputs=2, n_neurons=32),
    Layer(n_inputs=32, n_neurons=1),
])

networks = [
    (2, circle_net_1),
    (32, circle_net_2)
]

for hidden_size, network in networks:
    correct = 0
    outputs = []

    for inputs, expected in data:
        result = network.forward(inputs)[0]
        outputs.append(result)

        predicted = 1 if result >= 0.5 else 0

        if predicted == expected:
            correct += 1

    mean_output = sum(outputs) / len(outputs)

    print(f"\nHidden neurons: {hidden_size}")
    print(f"Accuracy: {correct}/{len(data)} "
          f"({100 * correct / len(data):.1f}%)")
    print(f"Minimum output: {min(outputs):.4f}")
    print(f"Maximum output: {max(outputs):.4f}")
    print(f"Mean output: {mean_output:.4f}")
