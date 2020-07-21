import os
import music21 as m21

KERN_DATASET_PATH = "./data/deutschl/test/"
ACCEPTABLE_DURATIONS = [
    0.25,
    0.5,
    0.75,
    1,
    1.5,
    2,
    3,
    4
]

SAVE_PREPROCESSING_DIR = "./data/dataset/"

def load_songs_in_kern(dataset_path):

    # Go through all the files in the dataset and load them with music21
    songs = []

    for path, subdirs, files in os.walk(dataset_path):
        for file_i in files:
            if file_i[-3:] == "krn":
                song = m21.converter.parse(os.path.join(path, file_i))
                songs.append(song)
    
    return songs

def has_acceptable_durations(song, acceptable_durations):
    
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    
    return True

def transpose(song):

    # Get the key from the song
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4] # The key on this data set is on the 4th eleeent

    # Estimate Key using music21
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")

    # Get the interval for transposition, E.G. Bmaj -> Cmaj
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    # Transpose song by calculated interval
    transposed_song = song.transpose(interval)

    return transposed_song

def encode_song(song, time_step=0.25):
    # p = 68, d = 1.0 -> [68, "_", "_", "_"]
    encoded_song = []

    for event in song.flat.notesAndRests:
        
        # Handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi # 68
        elif isinstance(event, m21.note.Rest):
            symbol = "r"
        
        # Convert the note/rest into time series notation
        steps = int(event.duration.quarterLength / time_step)
        for step_i in range(steps):
            if step_i == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")
        
    # Cast encoded song to a str
    encoded_song = " ".join(map(str, encoded_song))

    return encoded_song



def preprocess(dataset_path):

    # Load the folk songs
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")


    for i, song in enumerate(songs):

        # Filter out songs that have non-acceptable durations
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue
        # Transpose songs to Cmaj/Amin
        song = transpose(song)

        # Encode songs with music time series representation
        encoded_song = encode_song(song)

        # Save songs to text file
        save_path = os.path.join(
            SAVE_PREPROCESSING_DIR,
            str(i)
        )

        with open(save_path, "w") as fp:
            fp.write(encoded_song)


if __name__ == "__main__":

    songs = load_songs_in_kern(KERN_DATASET_PATH)
    print(F"Loaded {len(songs)} songs.")
    song = songs[0]
    
    preprocess(KERN_DATASET_PATH)

    # Transpose Song
    transposed_song = transpose(song)
    transposed_song.show()
