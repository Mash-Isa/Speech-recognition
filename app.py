import streamlit as st
import tensorflow as tf
import numpy as np
from scipy.io import wavfile
import sounddevice as sd
import tempfile
import os
from audio_recorder_streamlit import audio_recorder

# Load the saved model
model = tf.keras.models.load_model("model.h5")

st.title("Voice Command Recognition App")

# Function to record and process the voice command
def record_voice_command():
    st.title("Audio Recorder")
    st.info("Please speak your command...")
    audio = audio_recorder(pause_threshold=2.0, sample_rate=8000)

    if audio is not None:
        # To play audio in frontend:
        st.audio(audio)
        
        # To save audio to a file:
        wav_file = open("audio.mp3", "wb")
        wav_file.write(audio)
        return audio
    return None


def predict_voice_command(audio):
    # Preprocess the audio
    samples = audio #.flatten()
    if len(samples) == 8000:
        samples = samples.reshape(1, -1)

        # Make the prediction
        prediction = model.predict(samples)
        return prediction

# Capture and process voice command
# if st.button("Capture Voice Command"):
audio = record_voice_command()
if audio is not None:
    prediction = predict_voice_command(audio)

    # Get the class label based on the highest probability
    if prediction is not None:
        labels = ["bed", "bird", "cat", "dog", "down", "eight"]
        class_idx = np.argmax(prediction)
        predicted_class = labels[class_idx]

        st.success(f"Predicted Voice Command: {predicted_class}")
    else:
        st.warning("Failed to recognize the voice command. Please try again.")
