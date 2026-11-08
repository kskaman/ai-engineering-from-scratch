# Problem Statement 1 : Add a __sub__ method to the Value class 
# $(a - b = a + (-1 \cdot b))$. Then implement a __neg__ method. 
# Verify that the gradients are correct by comparing with manual 
# calculation for a simple expression like (a - b)^2.

# Problem Statement 2 : Add a relu method to Value (output max (0, x)
# , derivative is 1 if x > 0, else 0). Replace sigmoid with relu 
# in the hidden layers and train on XOR again. Compare 
# convergence speed. You should see faster training -- 
# this previews Lesson 04.

# Problem Statement 3 : Implement a __pow__ method on Value 
# for integer powers. Use it to replace mse_loss with a 
# proper (predicted - target) ** 2 expression. Verify 
# gradients match the original implementation.

# Problem Statement 4 : Add gradient clipping to the training 
# loop: after calling backward(), clip all gradients to [-1, 1]. 
# Train a deeper network (4+ layers with sigmoid) and compare 
# loss curves with and without clipping. This is your first 
# defense against exploding gradients.

# Problem Statement 5 : Build a visualization: after training 
# on XOR, print the gradient of every parameter in the 
# network. Identify which layer has the smallest gradients. 
# This demonstrates the vanishing gradient problem you read 
# about in the Concept section.

import math
import random


class Value:
    def __init__(self, data, children=(), op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._children = set(children)
        self._op = op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, exponent):
        if not isinstance(exponent, int):
            raise TypeError("The exponent must be an integer")
        out = Value(self.data ** exponent, (self,), f'**{exponent}')

        def _backward():
            if exponent != 0:
                self.grad += exponent * self.data ** (exponent - 1) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        x = max(-500, min(500, self.data))
        s = 1.0 / (1.0 + math.exp(-x))
        out = Value(s, (self,), 'sigmoid')

        def _backward():
            self.grad += (s * (1 - s)) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()


class Neuron:
    def __init__(self, n_inputs, activation='sigmoid'):
        if activation not in ('sigmoid', 'relu'):
            raise ValueError("Activation must be 'sigmoid' or 'relu'")
        self.activation = activation
        scale = (2.0 / n_inputs) ** 0.5
        self.weights = [Value(random.uniform(-scale, scale)) for _ in range(n_inputs)]
        self.bias = Value(0.0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.weights, x)), self.bias)
        return act.relu() if self.activation == 'relu' else act.sigmoid()

    def parameters(self):
        return self.weights + [self.bias]


class Layer:
    def __init__(self, n_inputs, n_outputs, activation='sigmoid'):
        self.neurons = [Neuron(n_inputs, activation) for _ in range(n_outputs)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        params = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params


class Network:
    def __init__(self, sizes, hidden_activation='sigmoid'):
        self.layers = []
        for i in range(len(sizes) - 1):
            activation = hidden_activation if i < len(sizes) - 2 else 'sigmoid'
            self.layers.append(Layer(sizes[i], sizes[i + 1], activation))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            if not isinstance(x, list):
                x = [x]
        return x[0] if len(x) == 1 else x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0


    def loss(self, data):
        return sum(((self(inputs) - target) ** 2 for inputs, target in data), Value(0.0))

    def train(self, data, epochs=1000, learning_rate=1.0, clip_gradients=False):
        data = list(data)
        losses = []
        for epoch in range(epochs):
            total_loss = self.loss(data)
            losses.append(total_loss.data)
            self.zero_grad()
            total_loss.backward()
            for parameter in self.parameters():
                if clip_gradients:
                    parameter.grad = max(-1.0, min(1.0, parameter.grad))
                parameter.data -= learning_rate * parameter.grad

        self.zero_grad()
        self.loss(data).backward()
        return losses

    def print_gradients(self):
        layer_means = []
        for layer_index, layer in enumerate(self.layers, start=1):
            for neuron_index, neuron in enumerate(layer.neurons, start=1):
                for weight_index, weight in enumerate(neuron.weights, start=1):
                    print(f"Layer {layer_index}, neuron {neuron_index}, "
                          f"weight {weight_index}: {weight.grad:.6e}")
                print(f"Layer {layer_index}, neuron {neuron_index}, "
                      f"bias: {neuron.bias.grad:.6e}")
            parameters = layer.parameters()
            mean_gradient = sum(abs(parameter.grad) for parameter in parameters) / len(parameters)
            layer_means.append(mean_gradient)
            print(f"Layer {layer_index} mean absolute gradient: {mean_gradient:.6e}")

        smallest_layer = min(range(len(layer_means)), key=layer_means.__getitem__) + 1
        print(f"Smallest mean absolute gradient: layer {smallest_layer}")
        return smallest_layer
