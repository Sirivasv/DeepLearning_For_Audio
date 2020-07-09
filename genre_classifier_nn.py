import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATASET_PATH = "./data.json"

def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)
    
    # Convert lists into numpy arrays
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets

if __name__ == "__main__":
    
    # Load Data
    inputs, targets = load_data(DATASET_PATH)

    # Split the data into train and test sets
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(
        inputs, targets, test_size=0.3)
    
    # Build the network architecture
    model = keras.Sequential([
        # input layer
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),

        # 1st hidden layer
        keras.layers.Dense(512, activation="relu"),

        # 2nd hidden layer
        keras.layers.Dense(256, activation="relu"),

        # 3rd hidden layer
        keras.layers.Dense(64, activation="relu"),

        # output layer
        keras.layers.Dense(10, activation="softmax")

    ])

    # Compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    model.summary()

    # Train network
    model.fit(
        inputs_train, targets_train, 
        validation_data=(
            inputs_test, targets_test
        ),
        epochs=50,
        batch_size=32
    )

