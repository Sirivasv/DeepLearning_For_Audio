import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATA_PATH = "./data.json"
full_data = []

def load_data(data_path):
    """
    Load training dataset from json file.
    :param data_path (str): Path to json file containing data
    :return X (ndarray): Inputs
    :return y (ndarray): Targets
    """
    global full_data

    with open(data_path, "r") as fp:
        data = json.load(fp)
        full_data = data
    
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y

def prepare_datasets(test_size, validation_size):

    # Load the data
    X, y = load_data(DATA_PATH)

    # Create Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size)

    # Create Train/Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=validation_size)
    
    # 3d array -> (130, 13, 1)
    X_train = X_train[..., np.newaxis] # 4d array (n_samples, 130, 13, 1)
    X_val = X_val[..., np.newaxis] # 4d array (n_samples, 130, 13, 1)
    X_test = X_test[..., np.newaxis] # 4d array (n_samples, 130, 13, 1)

    return X_train, X_val, X_test, y_train, y_val, y_test

def build_model(input_shape):
    # Create a model
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

def predict(model, X, y):

    X = X[np.newaxis, ...]
    y_pred = model.predict(X) # X -> (1, 130, 13, 1)

    # prediction = [ [0.1, 0.2, ...]]
    predicted_index = np.argmax(y_pred, axis=1) # [4]

    print("Expected index: {}, Predicted index: {}".format(
        y, predicted_index))
    print("Expected label: {}, Predicted label: {}".format(
        full_data["mapping"][y], full_data["mapping"][predicted_index[0]]))

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

    # Train the CNN
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=32, epochs=30
    )

    # Evaluate the CNN on the test set
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))

    # Make prediction on a sample
    X = X_test[100]
    y = y_test[100]

    predict(model, X, y)