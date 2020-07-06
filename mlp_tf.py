import numpy as np
import tensorflow as tf
from random import random
from sklearn.model_selection import train_test_split

# array[[0.1, 0.2], [0.2, 0.2]]
# array[[0.3], [0.4]]

def generate_dataset(num_samples, test_size):
    # Create dummy data
    inputs = np.array([
        [random() / 2 for _ in range(2)] for _ in range(num_samples)
    ])
    targets = np.array([
        [i[0] + i[1]] for i in inputs
    ])

    X_train, X_test, y_train, y_test = train_test_split(inputs,
        targets, test_size=test_size)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":

    X_train, X_test, y_train, y_test = generate_dataset(5000, 0.3)

    #, X_test, y_train, y_test

    # print("X_test: \n {}".format(X_test))
    # print("y_test: \n {}".format(y_test))
    
    # build model: 2 -> 5 -> 1
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(5, input_dim=2, activation="sigmoid"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # compile model
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    model.compile(optimizer=optimizer, loss="MSE")

    # train model
    model.fit(X_train, y_train, epochs=15, batch_size=1)

    # evaluate model
    print("\nModel evaluation:")
    model.evaluate(X_test, y_test, batch_size=1)

    # make predictions
    data = np.array([[0.1, 0.2], [0.2, 0.2]])
    predictions = model.predict(data)

    print("\nSome predictions:")
    for d, p in zip(data, predictions):
        print("{} + {} = {}".format(d[0], d[1], p[0]))
