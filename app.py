import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="Genre Classifier", page_icon="🎧", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0F0E1C; color: #EDEDED; }
    div.stButton > button { background-color: #C139D1; color: #EDEDED; border-radius: 10px; border: none; }
    div.stButton > button:hover { background-color: #9629A2; }
    div[data-testid="stFileUploader"] { background-color: #151520; border: 1px solid #2E2E3E; border-radius: 15px; padding: 20px; }
    h1, h2, h3 { color: #EDEDED; }
    p { color: #EDEDED; }
</style>
""", unsafe_allow_html=True)

SAMPLE_RATE = 22050
DURATION = 30 
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

@st.cache_resource
def load_classifier():
    try:
        return tf.keras.models.load_model('audio_classifier.keras')
    except:
        return None

model = load_classifier()

def process_audio(file_path):
    # load audio file
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    
    # extract MFCCs
    n_mfcc = 13
    n_fft = 2048
    hop_length = 512
    num_segments = 10 
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    
    # take the middle slice of the audio for analysis
    start = int(len(signal) / 2)
    end = start + samples_per_segment
    slice_signal = signal[start:end]

    mfcc = librosa.feature.mfcc(y=slice_signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc = mfcc.T
    
    # reshape for model input (1, time_steps, n_mfcc, 1)
    mfcc_input = mfcc[np.newaxis, ..., np.newaxis]
    
    return mfcc_input, mfcc 

st.title("Genre Classifier")
st.write("Upload a 30-second song clip (WAV/MP3). The AI will visualize the sound and predict the genre.")

if model is None:
    st.error("Model not found! Make sure 'audio_classifier.keras' is in the same folder.")
    st.stop()

uploaded_file = st.file_uploader("Drop your track here", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save temp file
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(uploaded_file)
    
    left_spacer, center_col, right_spacer = st.columns([1, 1, 1])
    with center_col:
        analyze_btn = st.button("Analyze Audio", use_container_width=True)

    if analyze_btn:
        with st.spinner("Listening to the track..."):
            try:
                # process
                X, mfcc_viz = process_audio("temp_audio.wav")
                
                # predict
                prediction = model.predict(X)
                predicted_index = np.argmax(prediction, axis=1)[0]
                confidence = np.max(prediction)
                predicted_genre = GENRES[predicted_index]

                st.divider() # Adds a nice horizontal line

                # main results
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.success(f"Top Prediction: **{predicted_genre.upper()}**")
                with res_col2:
                    st.metric("Confidence Level", f"{confidence*100:.1f}%")

                st.write("") # spacer

                # insights
                st.subheader("AI Analysis Deep Dive")
                tab1, tab2 = st.tabs(["Confidence Breakdown", "What the AI Sees (MFCC)"])

                with tab1:
                    # Create a dictionary for the chart
                    probs = prediction[0]
                    chart_data = dict(zip(GENRES, probs))
                    sorted_data = dict(sorted(chart_data.items(), key=lambda item: item[1], reverse=True))
                    
                    st.bar_chart(sorted_data)

                with tab2:
                    st.caption("This heatmap represents the 'texture' of the sound. Fast, loud genres look dense and bright, while acoustic genres look sparse.")
                    
                    # Plot the MFCC
                    fig, ax = plt.subplots(figsize=(10, 3))
                    fig.patch.set_facecolor('#0F0E1C')
                    ax.set_facecolor('#0F0E1C')
                    
                    img = librosa.display.specshow(mfcc_viz.T, x_axis='time', ax=ax, cmap='coolwarm')
                    cbar = plt.colorbar(img, ax=ax, format="%+2.0f dB")
                    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
                    
                    # Stylize axes
                    ax.xaxis.label.set_color('white')
                    ax.yaxis.label.set_color('white')
                    ax.tick_params(axis='x', colors='white')
                    ax.tick_params(axis='y', colors='white')
                    
                    st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"Error processing audio: {e}")
            finally:
                # cleanup
                if os.path.exists("temp_audio.wav"):
                    os.remove("temp_audio.wav")