import numpy as np

class Perceptron(object):

    def __init__(self, n_inputs, max_epochs=100, learning_rate=1e-2):
        """
        Initializes the perceptron object.
        - n_inputs: Number of inputs.
        - max_epochs: Maximum number of training cycles.
        - learning_rate: Magnitude of weight changes at each training cycle.
        - weights: Initialize weights (including bias).
        """
        self.n_inputs =  n_inputs # Fill in: Initialize number of inputs
        self.max_epochs = max_epochs  # Fill in: Initialize maximum number of epochs
        self.learning_rate = learning_rate  # Fill in: Initialize learning rate
        self.weights = np.random.rand(n_inputs + 1) * 0.01  # Fill in: Initialize weights with zeros 
        
    def forward(self, input_vec):
        """
        Predicts label from input.
        Args:
            input_vec (np.ndarray): Input array of training data, input vec must be all samples
        Returns:
            int: Predicted label (1 or -1) or Predicted lables.
        """
        # Add bias term to the input (always 1 for bias)
        input_with_bias = np.append(input_vec, 1)  # Append 1 for bias
        # Compute dot product of inputs and weights
        weighted_sum = np.dot(input_with_bias, self.weights)
        # Apply the step activation function to get binary output (-1 or 1)
        return 1 if weighted_sum >= 0 else -1


        
    def train(self, training_inputs, labels):
        """
        Trains the perceptron.
        Args:
            training_inputs (list of np.ndarray): List of numpy arrays of training points.
            labels (np.ndarray): Array of expected output values for the corresponding point in training_inputs.
        """
        # we need max_epochs to train our model
        for _ in range(self.max_epochs): 
            """
                What we should do in one epoch ? 
                you are required to write code for 
                1.do forward pass
                2.calculate the error
                3.compute parameters' gradient 
                4.Using gradient descent method to update parameters(not Stochastic gradient descent!,
                please follow the algorithm procedure in "perceptron_tutorial.pdf".)
            """
            global_error = 0  # Track total errors for convergence
            
            for input_vec, label in zip(training_inputs, labels):
                prediction = self.forward(input_vec)  # Forward pass
                error = label - prediction  # Calculate the error (expected - predicted)

                # If there is an error, update the weights
                if error != 0:
                    input_with_bias = np.append(input_vec, 1)  # Append bias term to input
                    # Update weights: w = w + eta * error * input
                    self.weights += self.learning_rate * error * input_with_bias
                    global_error += 1  # Count the error

            # If no errors in the entire epoch, we can stop early (converged)
            if global_error == 0:
                print(f"Training converged after {_+1} epochs")
                break
        else:
            print(f"Training completed after {self.max_epochs} epochs")


        
