import librosa
import os
import json

DATASET_PATH = "dataset/"
JSON_PATH = "data.json"
SAMPLES_TO_CONSIDER = 22050 # 1s worth of sound

def prepare_dataset(
        dataset_path, json_path, n_mfcc=13, hop_length=512, n_fft=2048):

    # Create data dictionary
    data = {
        "mappings": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    # Loop through all the subdirs

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        # We need to ensure that we're not at root level
        if dirpath is not dataset_path:
            
            # Update mappings
            category = dirpath.split("/")[-1]
            data["mappings"].append(category)
            print(f"Processing {category}")

            # Loop through all the filenames and extract MFCCs
            for f in filenames:

                # Get the file path
                file_path = os.path.join(dirpath, f)

                # Load audio file
                signal, sr = librosa.load(file_path)

                # Ensure Audio file is at least 1 sec
                if len(signal) >= SAMPLES_TO_CONSIDER:

                    # Enforce 1 sec. long signals
                    signal[:SAMPLES_TO_CONSIDER]

                    # Extract the MFCCs
                    MFCCs = librosa.feature.mfcc(
                        signal,
                        n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
                    
                    # Store data
                    data["labels"].append(i-1)
                    data["MFCCs"].append(MFCCs.tolist())
                    data["files"].append(file_path)
                    print(f"{file_path}:{i-1}")

    # Store in json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
    
if __name__ == "__main__":
    
    prepare_dataset(DATASET_PATH, JSON_PATH)
