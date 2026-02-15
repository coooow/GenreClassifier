import json
import math
import librosa
import os


DATASET_PATH = './archive/Data/genres_original'
JSON_PATH = './data.json'
SAMPLE_RATE = 22050
DURATION = 30  # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    data = {
        "mapping": [], # genre labels
        "mfcc": [], # input features
        "labels": [] # target labels
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments) # 5 segments per track
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)
    
    print("Reading dataset, this may take a while...")
    
    # go through all genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path: # checking if we are not at root
            dir_components = dirpath.split("/") # get genre label
            semantic_label = dir_components[-1]
            data["mapping"].append(semantic_label)
            print(f"Processing: {semantic_label}") # processing said genre
            
            # go through all audio files in genre sub-dir
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                
                if not file_path.endswith(".wav"): # only process .wav files
                    continue
                
                try:
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                    
                    # process segments extracting mfcc and storing data
                    for s in range(num_segments):
                        start_sample = num_samples_per_segment * s
                        finish_sample = start_sample + num_samples_per_segment
                        
                        mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample], 
                                                    sr=sr, 
                                                    n_fft=n_fft, 
                                                    n_mfcc=n_mfcc, 
                                                    hop_length=hop_length)
                        
                        mfcc = mfcc.T # transpose to get time steps as rows
                        
                        # store only mfcc feature with expected number of vectors
                        if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(i-1) # label index (i-1 because i starts at 1 for genres)
                            print(f"{file_path}, segment:{s+1}") # log file and segment number
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    
    # save data to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
    
    print("Data saved to", json_path)