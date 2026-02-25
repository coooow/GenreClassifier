# 🎧 Genre Classifier

A deep learning system that uses **Computer Vision** and **Convolutional Neural Networks (CNNs)** to accurately classify music genres by "seeing" sound.

## Overview
Standard audio analysis struggles with raw sound waves because they contain too much noise and unhelpful data. 

This project solves that by treating audio classification as an **Image Recognition** problem:
1.  **Feature Extraction:** Converts raw `.wav` or `.mp3` files into Mel-Frequency Cepstral Coefficients (MFCCs)—essentially "heatmaps" of sound texture that mimic human hearing.
2.  **Deep Learning:** Uses a custom-trained CNN to scan these images for visual patterns (e.g., the bright, dense blocks of Heavy Metal vs. the sparse lines of Classical).
3.  **Probabilistic Output:** Processes the features through dense layers to return a confidence breakdown across 10 distinct genres.

## Tech Stack
* **Core:** Python 3.10+
* **Deep Learning:** TensorFlow / Keras, NumPy
* **Audio Processing:** Librosa
* **Frontend:** Streamlit, Matplotlib (for real-time spectrogram rendering)
* **Data Source:** [GTZAN Genre Collection](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) (1000 audio tracks)

## Key Features
* **Audio-to-Image Pipeline:** Slices 30-second audio files and computes MFCCs to capture human-perceptible frequencies (Mel scale) while discarding background noise.
* **Custom CNN Architecture:** Implements multiple `Conv2D` and `MaxPooling2D` layers with `BatchNormalization` to detect spatial textures in the audio spectrograms.
* **Overfit Prevention:** Utilizes `Dropout` layers to force the neural network to rely on multiple audio cues rather than memorizing specific training tracks.
* **Interactive UI:** Features a built-in audio player, interactive confidence bar charts, and a real-time rendering of "What the AI sees" (the raw MFCC heatmap).

## How to Run / View

**Option 1: Live Demo (Recommended)**

This project is hosted on Streamlit. You can access the live version here:

**➡️ [Click to view Live Demo](https://coooow-genreclassifier-app-vhlcp3.streamlit.app/)**

**Option 2: Run Locally**

If you want to explore the model on your own machine:

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/coooow/GenreClassifier.git](https://github.com/coooow/GenreClassifier.git)
    cd AudioClassifier
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**
    *(Make sure the `audio_classifier.keras` model file is in the root directory)*
    ```bash
    streamlit run app.py
    ```
