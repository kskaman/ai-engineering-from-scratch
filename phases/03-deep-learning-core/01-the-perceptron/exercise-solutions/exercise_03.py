# Problem Statement : Build a 3-input perceptron that 
# outputs 1 only when at least 2 of the 3 inputs are 1 
# (a majority vote function). Is this linearly separable? Why?

# Yes, the three-input majority function is linearly
# separable.
# Since there are three binary inputs, the eight 
# possible input combinations can be represented as 
# vertices of a cube in three-dimensional space.
#
# The function outputs 1 when at least two inputs are
#  1, meaning the sum of the inputs is greater than 
# or equal to 2. Otherwise, it outputs 0.
#
# A plane defined by:
#
# [
# x_1+x_2+x_3=1.5
# ]
#
# separates the two classes. Points with an input sum 
# of 0 or 1 lie on one side of the plane, while points 
# with an input sum of 2 or 3 lie on the other.
#
# Therefore, a single plane can separate all the positive and negative examples, making the majority function linearly separable.


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


three_input_data = [
    ([0, 0, 0], 0),
    ([0, 0, 1], 0),
    ([0, 1, 0], 0),
    ([0, 1, 1], 1),
    ([1, 0, 0], 0),
    ([1, 0, 1], 1),
    ([1, 1, 0], 1),
    ([1, 1, 1], 1)
]

test_gate("Three-Input Perceptron", 3, three_input_data)