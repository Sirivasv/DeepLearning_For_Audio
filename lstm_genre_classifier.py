import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

DATA_PATH = "./data.json"

def load_data(data_path):
    """Loads training dataset from json file.

        :param data_path (str): Path to json file containing data
        :return x (ndarray): Inputs
        :return y (ndarray): Targets
    """
    
    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y

def plot_history(history):
    """ Plots accuracy/loss for training/validation set
        as a function of the epochs.
        
        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # Create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy Eval")

    # Create error subplot
    axs[1].plot(history.history["error"], label="train error")
    axs[1].plot(history.history["val_error"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error Eval")

    plt.show()

def prepare_datasets(test_size, validation_size):
    """Loads data and splits it into train, validation and test sets.

    :param test_size (float): Value in [0, 1] indicating percentage of data
        set to allocate to test split
    :param validation_size (float): Value in [0, 1] indicating percentage of
        train set to allocate to validation split

    :return X_train (ndarray): Input training set
    :return X_validation (ndarray): Input validation set
    :return X_test (ndarray): Input test set
    :return y_train (ndarray): Target training set
    :return y_validation (ndarray): Target validation set
    :return y_test (ndarray): Target test set
    """

    # load data
    X, y = load_data(DATA_PATH)

    # Create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=validation_size)
    
    # Add an axis to input sets
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_model(input_shape):
    """Generates CNN model
    
    :param input_shape (tuple): Shape of input set
    :return model: CNN model
    """

    # build network topology
    model = keras.Sequential()

    # 1st Conv Layer
    model.add(keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=input_shape
    ))
    model.add(keras.layers.MaxPool2D(
        pool_size=(3, 3),
        strides=(2,2),
        padding="same"
    ))
    model.add(keras.layers.BatchNormalization())

    # 2nd Conv Layer
    model.add(keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=input_shape
    ))
    model.add(keras.layers.MaxPool2D(
        pool_size=(3, 3),
        strides=(2,2),
        padding="same"
    ))
    model.add(keras.layers.BatchNormalization())
    # 3rd Conv Layer
    model.add(keras.layers.Conv2D(
        filters=32,
        kernel_size=(2, 2),
        activation="relu",
        input_shape=input_shape
    ))
    model.add(keras.layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2,2),
        padding="same"
    ))
    model.add(keras.layers.BatchNormalization())

    # Flatten the output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(
        units=64,
        activation="relu"
    ))
    model.add(keras.layers.Dropout(0.3))

    # Output layer
    model.add(keras.layers.Dense(
        units=10, activation="softmax"
    ))

    return model

if __name__ == "__main__":

    # Create train, validation and test set
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_datasets(0.25,
                                                                      0.2)
    # Build the CNN net
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape)

    # Compile the Network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # Train the CNN
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=32, epochs=30
    )

    # Plot the accuracy and error over the epochs
    plot_history(history)

    # Evaluate the CNN on the test set
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
    print("Accuracy on test set is: {}".format(test_accuracy))