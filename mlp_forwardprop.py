import numpy as np
from random import random

# Save the activations and derivatives
# Implement Backpropagation
# Implement Gradient descent
# Implement Train method
# Train our net with some dummy dataset
# Make some predictions

class MLP(object):
    """A Multilayer Perceptron class
    """
    
    def __init__(self, num_inputs=3, hidden_layers=[3, 3], num_outputs=2):
        """Constructor for the MLP. Takes the number of inputs,
            a variable number of hidden layers, and a number of outputs
        Args:
            num_inputs (int): Number of inputs
            hidden_layers (list): A list of ints for the hidden layers
            num_outputs (int): Number of outputs
        """
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        # Create a generic representation of the layers
        layers = [self.num_inputs] + self.hidden_layers + [self.num_outputs]

        # Create random connection wights for the layers
        self.weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            self.weights.append(w)

        # Create zero-valued activations
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

        # Create random derivatives
        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives

        
    def forward_propagate(self, inputs):
        """Computes forward propagation of the network based on input signals

        Args:
            inputs (ndarray): Input signals
        Returns:
            activations (ndarray): Output values
        """

        # The input layer activation is just the input itself
        activations = inputs
        self.activations[0] = inputs

        # Iterate through the network layers
        for i, w in enumerate(self.weights):
            # Calculate matrix multiplication between previous
            # activation and weight matrix
            net_inputs = np.dot(activations, w)

            # Apply sigmoid activation function
            activations = self._sigmoid(net_inputs)
            self.activations[i+1] = activations

        # return output layer activation
        return activations
    
    def back_propagate(self, error, verbose=False):
        """Calculates and pass backwards the derivatives

        Args:
            error (float): The error in the prediction
            verbose (boolean): Flag to print derivatives per layer
        Returns:
            error (float): Last calculated error
        """
        
        # dE/dW_i = (y - a_[i+1]) s'(h_[i+1])a_i
        # s'(h_[i+1]) = s(h_[i+1])(1 - s(h_[i+1]))
        # s(h_[i+1]) = a_[i+1] 

        # dE/W_[i-1] = (y - a_[i+1]) s'(h_[i+1]) W_i s'(h_i) a_[i-1]

        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]

            delta = error * self._sigmoid_derivative(activations)
            # ndarray([0.1, 0.2]) --> ndarray([[0.1, 0.2]])
            delta_reshaped = delta.reshape(delta.shape[0], -1).T

            current_activations = self.activations[i] 
            # ndarray([0.1, 0.2]) --> ndarray([[0.1], [0.2]])
            current_activations_reshaped = current_activations.reshape(
                current_activations.shape[0], -1)

            self.derivatives[i] = np.dot(current_activations_reshaped,
                delta_reshaped)
            error = np.dot(delta, self.weights[i].T)

            if verbose:
                print("Derivatives for W{}: {}".format(
                    i,
                    self.derivatives[i]))
        
        return error

    def gradient_descent(self, learning_rate):
        """Modify the weights in the network using
            the derivatives of each layer

        Args:
            learning_rate (float): The rate for each modification
        """

        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learning_rate

    def train(self, inputs, targets, epochs, learning_rate):
        """Run Gradient Descent to modify the weights according to 
            target values

        Args:
            inputs (ndarray): List of lists each containing the values 
                of the inputs per data sample 
            targets (ndarray): List of lists each 
                with the ground truth per data sample
            epochs (int): Number of training steps to run
            learning_rate (float): rate of wight modification
        """
        for i in range(epochs):
            sum_error = 0
            for input, target in zip(inputs, targets):
                # Forward propagation
                output = self.forward_propagate(input)

                # Calculate the error
                error = target - output

                # Back propagation
                self.back_propagate(error)

                # Apply gradient descent
                self.gradient_descent(learning_rate=learning_rate)

                # Sum error
                sum_error += self._mse(target, output)

            # Report error
            print("Error: {} at epoch: {}".format(sum_error/len(inputs), i))

    def _mse(self, target, output):
        """Calculates the Mean Squared Error
        
        Args:
            target (ndarray): list of one element, the target values
            output (ndarray): list of one element, the ouput of the network
        Returns:
            mse (float): the Mean Squared Error
        """
        return np.average((target-output)**2)

    def _sigmoid_derivative(self, x):
        """Calculate the first derivative of the sigmoid function assuming x
            is the value of the sigmoid function
        
        Args: 
            x (float): The result of sigmoid function
        Returns:
            y_1d (float): The value of the first derivative of the sigmoid
                function
        """
        y_1d = (x * (1.0 - x))
        return y_1d

    def _sigmoid(self, x):
        """Calculates the sigmoid evaluation of a float value x

        Args:
            x (float): the input for the sigmoid function
        Returns:
            y (float): the result of applying the sigmoid function to x
        """

        # Save the sigmoid value to y
        y = 1 / (1 + np.exp(-x))

        # Return y
        return y

if __name__ == "__main__":

    # Create an MLP
    mlp = MLP(
        num_inputs=2,
        hidden_layers=[3, 2],
        num_outputs=1)

    # Create dummy data
    inputs = np.array([
        [random() / 2 for _ in range(2)] for _ in range(1000)
    ])
    targets = np.array([
        [i[0] + i[1]] for i in inputs
    ])

    # Train an mlp
    mlp.train(inputs, targets, 200, 0.5)

    # Create dummy data
    input = np.array([0.4, 0.4])
    target = np.array([0.8])

    output = mlp.forward_propagate(input)
    print()
    print()
    print("Our network believes that {} + {} is equal to {}".format(input[0], input[1], output[0]))