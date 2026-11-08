# Problem Statement: Modify the Perceptron class to track 
# the decision boundary (w1x1 + w2x2 + b = 0) at each epoch. 
# Print how the line shifts during training on the AND gate.

class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1):
        self.weights = [0.0] * n_inputs
        self.bias = 0.0
        self.lr = learning_rate

    def predict(self, inputs):
        total = sum(w * x for w, x in zip(self.weights, inputs))
        total += self.bias
        return 1 if total >= 0 else 0

    def train(self, training_data, epochs=100):
        for epoch in range(epochs):
            errors = 0
            for inputs, target in training_data:
                prediction = self.predict(inputs)
                error = target - prediction
                if error != 0:
                    errors += 1
                    for i in range(len(self.weights)):
                        self.weights[i] += self.lr * error * inputs[i]
                    self.bias += self.lr * error
            
            print(
                f"Epoch {epoch + 1}: "
                f"{self.weights[0]:.2f}x1 + "
                f"{self.weights[1]:.2f}x2 + "
                f"{self.bias:.2f} = 0"
            )
            
            if errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                return
        print(f"Did not converge after {epochs} epochs")


def test_gate(name, n_inputs, data):
    print(f"=== {name} ===")
    p = Perceptron(n_inputs)
    p.train(data)
    print(f"  Weights: {p.weights}, Bias: {p.bias}")
    for inputs, expected in data:
        result = p.predict(inputs)
        status = "OK" if result == expected else "WRONG"
        print(f"  {inputs} -> {result} (expected {expected}) {status}")
    print()

    for inputs, expected in data:
        z = sum(w * x for w, x in zip(p.weights, inputs)) + p.bias

        if expected == 1:
            assert z >= 0, f"Invalid positive classification: {inputs}"
        else:
            assert z < 0, f"Invalid negative classification: {inputs}"

    print("Decision boundary verified!")

and_data = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1)
]

test_gate("AND Perceptron", 2, and_data)