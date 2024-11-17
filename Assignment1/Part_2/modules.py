import numpy as np

class Linear(object):
    def __init__(self, in_features, out_features):
        """
        Initializes a linear (fully connected) layer. 
        TODO: Initialize weights and biases.
        - Weights should be initialized to small random values (e.g., using a normal distribution).
        - Biases should be initialized to zeros.
        Formula: output = x * weight + bias
        """
        # Initialize weights and biases with the correct shapes.
        self.params = {
            'weight': np.random.randn(in_features, out_features) * 0.01,
            'bias': np.zeros(out_features)
        }
        self.grads = {
            'weight': np.zeros_like(self.params['weight']),
            'bias': np.zeros_like(self.params['bias'])
        }
        self.x = None

    def forward(self, x):
        """
        Performs the forward pass using the formula: output = xW + b
        TODO: Implement the forward pass.
        """
        self.x = x  # Storing input for backward pass
        return np.dot(x, self.params['weight']) + self.params['bias']


    def backward(self, dout):
        """
        Backward pass to calculate gradients of loss w.r.t. weights and inputs.
        TODO: Implement the backward pass.
        """
        self.grads['weight'] = np.dot(self.x.T, dout)  # Gradient w.r.t. weights
        self.grads['bias'] = np.sum(dout, axis=0)  # Gradient w.r.t. bias
        dx = np.dot(dout, self.params['weight'].T)  # Gradient w.r.t. input
        return dx


class ReLU(object):
    def forward(self, x):
        """
        Applies the ReLU activation function element-wise to the input.
        Formula: output = max(0, x)
        TODO: Implement the forward pass.
        """
        self.x = x
        return np.maximum(0, x)

    def backward(self, dout):
        """
        Computes the gradient of the ReLU function.
        TODO: Implement the backward pass.
        Hint: Gradient is 1 for x > 0, otherwise 0.
        """
        dx = dout.copy()
        dx[self.x <= 0] = 0  # Gradient is 0 for x <= 0
        return dx

class SoftMax(object):
    def forward(self, x):
        """
        Applies the softmax function to the input to obtain output probabilities.
        Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
        TODO: Implement the forward pass using the Max Trick for numerical stability.
        """
        shift_x = x - np.max(x, axis=1, keepdims=True)  # Max trick for numerical stability
        exp_x = np.exp(shift_x)
        self.out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.out

    def backward(self, dout):
        """
        The backward pass for softmax is often directly integrated with CrossEntropy for simplicity.
        TODO: Keep this in mind when implementing CrossEntropy's backward method.
        """
        return dout  # This layer typically passes through the gradient when used with CrossEntropy


class CrossEntropy(object):
    def forward(self, x, y):
        """
        Computes the CrossEntropy loss between predictions and true labels.
        Formula: L = -sum(y_i * log(p_i)), where p is the softmax probability of the correct class y.
        TODO: Implement the forward pass.
        """
        m = x.shape[0]
        self.probs = SoftMax().forward(x)
        self.y_true = y
        log_likelihood = -np.log(self.probs[np.arange(m), y])
        loss = np.sum(log_likelihood) / m
        return loss

    def backward(self, x, y):
        """
        Computes the gradient of CrossEntropy loss with respect to the input.
        TODO: Implement the backward pass.
        Hint: For softmax output followed by cross-entropy loss, the gradient simplifies to: p - y.
        """
        m = x.shape[0]
        dx = self.probs
        dx[np.arange(m), y] -= 1  # p - y
        return dx / m
