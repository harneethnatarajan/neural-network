# Neural Network From Scratch

A neural network library built PURELY in NumPy. No TensorFlow, no PyTorch, no frameworks.

## Features

- Dense (fully connected) layers
- ReLU and tanh activations
- Mean squared error calculations
- Backpropagation and gradient descent

## Usage

```
pip install numpy
```

```python
import numpy as np
from neuralnetwork import reluNetwork

# XOR test

# Each COLUMN is one sample: shape (features, samples)
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])
Y = np.array([[0, 1, 1, 0]])      # XOR answers for the four columns

# 2 inputs -> 4 hidden neurons (ReLU) -> 1 output
network = reluNetwork(inputSize=2, hiddenSize=4, outputSize=1)

for epoch in range(10000):
    network.forwardPropagation(X)                      # 1. make predictions
    network.backwardPropagation(Y, learningRate=0.01)  # 2. measure the error, nudge the weights

# Ask it about a new input: [1, 0] should give something close to 1
print(network.forwardPropagation(np.array([[1], [0]])))
```

## How it works

Every layer knows how to do two things: pass data forward, and pass an error backward.

- **Forward:** the layer takes its input `x` and computes `z = W·x + b`, where `W` are the weights and `b` is the bias (the numbers the network learns). An activation function `f` then squashes `z` into the output `a = f(z)`. That output becomes the input of the next layer, until you reach a prediction.
- **Backward (chain rule):** the error flows backwards through the layers. Each layer takes in the error at its output and hands back the error at its input, which is what the layer behind it receives. Here `L` is the loss (one number for how wrong the network is), and `dL/dz` means "how much the loss changes if `z` changes a tiny bit". A dense layer uses the incoming error `dL/dz` to work out how much each weight is to blame (`dL/dW = dL/dz · xᵀ`), and passes `dL/dx = Wᵀ · dL/dz` back to the previous layer. An activation layer just scales the error by its derivative.
- **Update:** each weight takes a small step against its blame, `W ← W − learningRate · dL/dW`, where `learningRate` controls the step size. The next prediction is then a bit less wrong.

Stack a few of these and you have a network that learns, or tries to atleast.

## References

- [Samson Zhang](https://www.youtube.com/watch?v=w8yWXqWQYmU)
