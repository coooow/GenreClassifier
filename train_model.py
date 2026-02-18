import json
import math
import librosa
import os

from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


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
                        
                        mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample], 
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
  
# uncomment the line to save the MFCC features from the dataset to a JSON file  
# save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)

def load_data(json_path):
    with open(json_path, "r") as fp:
        data = json.load(fp)
    
    X = data["mfcc"]
    y = data["labels"]
    
    return X, y

X, y = load_data(JSON_PATH)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X_train = X_train[..., np.newaxis] # add channel dimension for CNN input
X_test = X_test[..., np.newaxis]

input_shape = (X_train.shape[1], X_train.shape[2], 1) # (time steps, mfcc features, channels)

model = models.Sequential([
    
    # layer 1 - convolutional layer: extracts features from the MFCC input
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
    layers.BatchNormalization(),

    # layer 2 - convolutional layer: extracts more complex features
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
    layers.BatchNormalization(),

    # layer 3 - convolutional layer: extracts even more complex features
    layers.Conv2D(128, (2, 2), activation='relu'),
    layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'),
    layers.BatchNormalization(),

    layers.Flatten(),

    # dense layer
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3), # turns off 30% of neurons to prevent memorization

    # output layer
    layers.Dense(10, activation='softmax') 
])
    
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
                 