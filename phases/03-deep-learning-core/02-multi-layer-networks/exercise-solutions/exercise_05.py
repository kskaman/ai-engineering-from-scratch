# Problem Statement : Replace sigmoid with a "leaky step" 
# function: return 0.01 * z if z < 0, else 1.0. Run the 
# forward pass on XOR with the same hand-tuned weights from 
# Step 4. Does it still work? Why is the smooth sigmoid preferred 
# over hard cutoffs?

# Yes, the network still correctly classifies XOR using the 
# same hand-tuned weights and a threshold of 0.5. However, 
# the leaky step function produces negative outputs for some 
# inputs rather than values restricted to the range 0 to 1.
# Sigmoid is preferred over hard cutoff functions for gradient-based 
# learning because it is smooth and differentiable everywhere, allowing 
# gradients to propagate through the network during backpropagation. In
# contrast, the leaky step function has a zero derivative for positive 
# inputs and is not differentiable at zero, making weight updates 
# difficult. However, sigmoid can also suffer from vanishing gradients
# when its outputs saturate.
    
def leaky_step(z):
    return 0.01 * z if z < 0 else 1.0

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
            self.last_output.append(leaky_step(z))
        return self.last_output


class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        return current


hidden = Layer(
    n_inputs=2,
    n_neurons=2,
    weights=[[20.0, 20.0], [-20.0, -20.0]],
    biases=[-10.0, 30.0],
)

output = Layer(
    n_inputs=2,
    n_neurons=1,
    weights=[[20.0, 20.0]],
    biases=[-30.0],
)

xor_net = Network([hidden, output])

xor_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

for inputs, expected in xor_data:
    result = xor_net.forward(inputs)
    predicted = 1 if result[0] >= 0.5 else 0
    print(f"  {inputs} -> {result[0]:.6f} (rounded: {predicted}, expected: {expected})")