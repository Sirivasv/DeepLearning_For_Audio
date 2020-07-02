import numpy as np

class MLP:
    
    def __init__(self, num_inputs=3, num_hidden=[3, 5], num_outputs=2):
        
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]

        # initiate random weights
        self.weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            self.weights.append(w)
        
    def forward_propagate(self, inputs):
        activations = inputs

        for w in self.weights:
            # calculate net inputs
            net_inputs = np.dot(activations, w)

            # calculate the activations
            activations = self._sigmoid(net_inputs)

        return activations
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

if __name__ == "__main__":

    # create an MLP
    mlp = MLP(num_inputs=2, num_hidden=[3], num_outputs=1)

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
