import streamlit as st
import librosa
import numpy as np
import pickle
from audio_recorder_streamlit import audio_recorder
import io  # Add this import statement

# Load the trained model
with open('trained_model.pkl', 'rb') as file:
    etc = pickle.load(file)

# Define a function to extract features from an audio file
def extract_features(audio, sample_rate=44100):
    if isinstance(audio, bytes):
        audio_array, _ = librosa.load(io.BytesIO(audio), sr=sample_rate)
    elif isinstance(audio, str):
        audio_array, _ = librosa.load(audio, sr=sample_rate)
    elif isinstance(audio, np.ndarray):
        audio_array = audio
    else:
        raise ValueError("Unsupported audio data format")

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
    # Convert prediction to human-readable labels
    if prediction == 1:
        return "Human"
    elif prediction == 0:
        return "Deep Fake"

# Define the Streamlit app
def main():
    st.title("Voice Recognition Web App")
    
    # Option to record audio
    recording = audio_recorder()
    
    # Option to upload an audio file
    uploaded_file = st.file_uploader("Upload an audio file", type=['wav'])
    
    if recording:
        st.audio(recording)
        # Extract features and make prediction if recording is done
        if st.button("Classify recording"):
            prediction = classify_audio(recording)
            st.write("Prediction:", prediction)
    
    if uploaded_file is not None:
        # Check if the uploaded file is in WAV format
        if uploaded_file.type == 'audio/wav':
            # Extract features from the uploaded file and make prediction
            prediction = classify_audio(uploaded_file)
            st.write("Prediction:", prediction)
        else:
            st.error("Please upload a WAV file.")

    # Example Story
    st.write("---")
    st.header("Example Story for Testing")
    st.write("Once upon a time, in a quaint little town nestled between rolling hills and lush green forests, there lived a mischievous squirrel named Whiskers.")
    st.write("Whiskers was known throughout the town for his antics and adventures. Every day, he would scamper through the trees, searching for acorns and causing trouble wherever he went.")
    st.write("One sunny morning, as Whiskers was exploring the edge of the forest, he stumbled upon a hidden path leading deep into the woods. Curiosity piqued, he decided to follow it.")
    st.write("The path twisted and turned, leading Whiskers deeper into the heart of the forest than he had ever been before. Along the way, he encountered all sorts of creatures, from playful rabbits to wise old owls.")
    st.write("Finally, after what felt like hours of walking, Whiskers reached a clearing bathed in golden sunlight. In the center of the clearing stood a magnificent oak tree, its branches stretching high into the sky.")
    st.write("As Whiskers gazed up at the tree in wonder, he noticed something glinting in the sunlight. It was a golden acorn, shimmering with magic!")
    st.write("Without hesitation, Whiskers leaped up and grabbed the acorn, feeling a surge of energy coursing through his tiny body. Little did he know, this acorn held the key to a great adventure.")
    st.write("And so, with the golden acorn in his possession, Whiskers embarked on a journey that would take him to the far corners of the forest and beyond, encountering friends and foes alike along the way.")
    st.write("But that, dear reader, is a story for another day...")

if __name__ == "__main__":
    main()



