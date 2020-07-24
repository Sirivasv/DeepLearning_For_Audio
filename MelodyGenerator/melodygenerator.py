import json
import numpy as np
import music21 as m21
import tensorflow.keras as keras
from preprocessing import SEQUENCE_LENGTH, MAPPING_PATH

class MelodyGenerator():

    def __init__(self, model_path="model_1_256.h5"):

        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)
        
        self._start_symbols = ["/"] * SEQUENCE_LENGTH

    def generate_melody(self,
            seed,
            num_steps,
            max_sequence_length,
            temperature
            ):
        # seed "64 _ 63 _ _"
        
        # Create a seed with start symbols
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed

        # Map seed to int
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):

            # Limit the seed to max_sequence_legth
            seed = seed[-max_sequence_length:]

            # one-hot encode the seed
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
            # (max_sequence_length, num symbols in vocabulary) -->
            #   (1, max_sequence_length, num symbols in vocabulary)
            onehot_seed = onehot_seed[np.newaxis, ...]

            # Make a prediction
            probabilities = self.model.predict(onehot_seed)[0]
            # [0.1, 0.2, 0.1, 0.6]

            output_int = self._sample_with_temperature(
                probabilities, temperature)
            
            # Update the seed
            seed.append(output_int)

            # Map int to our encoding
            output_symbol = [k for k, v in self._mappings.items() \
                if v == output_int][0]
            
            # Check whether we're at the end of a melody
            if output_symbol == "/":
                break

            # Update the melody
            melody.append(output_symbol)

        return melody

    def _sample_with_temperature(self, probabilities, temperature):
        # temperature --> infinity // All random
        #   the distribution becomes irrelevant
        # temperature --> 0 // The one with
        #   the highest probability gets a probability of 1
        #   so it is deterministic
        # temperature = 1 // Sampling with the probabilities
        #   in the distribution, does not make 1 to the highest
        #   nor all to almost 0

        predictions = np.log(probabilities) / temperature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilities))
        index = np.random.choice(choices, p=probabilities)

        return index

    def save_melody(
            self,
            melody,
            step_duration=0.25,
            format="midi",
            file_name="mel.midi"):
        
        # Create a music21 Stream (using default key and time signature)
        stream = m21.stream.Stream()

        # Parse all the symbols in the melody and create note/rest objects
        # 60 _ _ _ r _ 62 _
        start_symbol = None
        step_counter = 1
        melody_length = len(melody)

        for i, symbol in enumerate(melody):

            # handle case in which we have a note/rest
            if (symbol != "_") or (i + 1 == melody_length):

                # ensure we're dealing with note/rest beyond the first one
                if start_symbol is not None:
                
                    quarter_length_duration = step_duration * step_counter

                    # handle rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)
                
                    # handle note
                    else:
                        m21_event = m21.note.Note(
                            int(start_symbol),
                            quarterLength=quarter_length_duration)
                    
                    # Append event
                    stream.append(m21_event)

                    # reset the step counter
                    step_counter = 1

                # update start symbol
                start_symbol = symbol

            # handle case in which we have a prolongation sign "_"
            else:
                step_counter += 1

        # Write the m21 stream to a midi file
        stream.write(format, file_name)

if __name__ == "__main__":

    mg = MelodyGenerator()
    seed = "55 _ _ _ 60 _ _ _ 55 _ _ _ 55 _"
    seed2 = "67 _ 67 _ 67 _ _ 65 64 _ 64 _ _"
    seed3 = "67 _ _ _ _ _ 65 _ 64 _ 62 _ 60 _ _ _"
    melody = mg.generate_melody(seed2, 500, SEQUENCE_LENGTH, 0.3)
    print(melody)
    print(len(melody))
    mg.save_melody(melody)
