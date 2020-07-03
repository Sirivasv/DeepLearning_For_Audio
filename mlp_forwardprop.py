import numpy as np

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
            # Calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations, w)

            # Apply sigmoid activation function
            activations = self._sigmoid(net_inputs)

        # return output layer activation
        return activations
    
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

    # create an MLP
    mlp = MLP(num_inputs=2, hidden_layers=[3], num_outputs=1)

    # create some inputs
    inputs = np.array([0.8, 1.0])

    # Modify wight matrices
    mlp.weights = np.array([
        np.array([
            np.array([1.2, 0.7, 1.0]),
            np.array([2, 0.6, 1.8])
        ]),
        np.array([
            np.array([1.0]),
            np.array([0.9]),
            np.array([1.5])
        ])
    ])

    # perform forward prop
    outputs = mlp.forward_propagate(inputs)

    # print the results
    print("The network input is: {}".format(inputs))
    print("The network output is: {}".format(outputs))
