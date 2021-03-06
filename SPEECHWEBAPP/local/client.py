import requests

URL = "http://52.204.96.173/predict"

TEST_AUDIO_FILE_PATH = "./classifier/dataset/left/1d919a90_nohash_1.wav"
# TEST_AUDIO_FILE_PATH = "./classifier/dataset/on/1aed7c6d_nohash_1.wav"

if __name__ == "__main__":
    
    audio_file = open(TEST_AUDIO_FILE_PATH, "rb")
    values = {"file": (TEST_AUDIO_FILE_PATH, audio_file, "audio/wav")}
    response = requests.post(URL, files=values)
    data = response.json()

    print(f"Predicted keyword is: {data['keyword']}")