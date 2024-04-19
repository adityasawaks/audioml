import streamlit as st
import librosa
import numpy as np
import pickle

# Load the trained model
with open('trained_model.pkl', 'rb') as file:
    etc = pickle.load(file)

# Define a function to extract features from an audio file
def extract_features(audio):
    sample_rate = 44100  # Sample rate for audio recorder
    audio_array = np.frombuffer(audio.getvalue(), dtype=np.int16)
    mfccs_features = librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=40)
    mfccs_features_mean = np.mean(mfccs_features.T, axis=0)
    return mfccs_features_mean

# Define a function to classify audio
def classify_audio(audio_data):
    # Extract features from the audio data
    audio_features = extract_features(audio_data)
    # Reshape features to 2D array
    audio_features = audio_features.reshape(1, -1)
    # Make prediction using the loaded model
    prediction = etc.predict(audio_features)
    return prediction

# Define the Streamlit app
def main():
    st.title("Voice Recognition Web App")
    
    # Option to record audio
    recording = st.audio_recorder("Record your audio", format="wav")
    
    # Option to upload an audio file
    uploaded_file = st.file_uploader("Upload an audio file", type=['wav'])
    
    if recording:
        st.audio(recording)
        # Extract features and make prediction if recording is done
        if st.button("Classify recording"):
            prediction = classify_audio(recording)
            st.write("Prediction:", prediction)
    
    if uploaded_file is not None:
        # Extract features from the uploaded file and make prediction
        prediction = classify_audio(uploaded_file)
        st.write("Prediction:", prediction)
        

if __name__ == "__main__":
    main()
