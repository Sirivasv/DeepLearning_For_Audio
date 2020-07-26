import tensorflow as tf
import tensorflow.keras as keras
from preprocessing import generate_training_sequences, SEQUENCE_LENGTH

OUTPUT_UNITS = 38
NUM_UNITS = [256]
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 64
SAVE_MODEL_PATH = "model_1_256.h5"

def build_model(output_units, num_units, loss, learning_rate):

    # Create the model architecture
    input = keras.layers.Input(shape=(None, output_units))
    x = keras.layers.LSTM(num_units[0])(input)
    x = keras.layers.Dropout(0.2)(x)

    output = keras.layers.Dense(output_units, activation="softmax")(x)

    model = keras.Model(input, output)

    # compile model
    model.compile(
        loss=loss,
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        metrics=["accuracy"]
    )

    model.summary()

    return model

def train(
        output_units=OUTPUT_UNITS,
        num_units=NUM_UNITS,
        loss=LOSS,
        learning_rate=LEARNING_RATE
        ):

    # Generate the train sequences
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

    # Build the network
    model = build_model(output_units, num_units, loss, learning_rate)

    # Train the model
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Save the model
    model.save(SAVE_MODEL_PATH)

if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    train()
