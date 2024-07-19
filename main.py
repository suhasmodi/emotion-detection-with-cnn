import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button
from keras.models import load_model
import librosa
import speech_recognition as sr

# Load the pre-trained model
model = load_model('SER_cnn_model.h5')

# Define the emotions
emotion_list = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Function to extract features from audio file
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# Function to predict emotion from audio file
def predict_emotion(file_path):
    features = extract_features(file_path)
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=2)
    prediction = model.predict(features)
    predicted_emotion = emotion_list[np.argmax(prediction)]
    return predicted_emotion

# Function to recognize text from audio file
def recognize_speech(file_path):
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(file_path)
    with audio_file as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        text = "Speech could not be understood"
    except sr.RequestError:
        text = "Could not request results; check your network connection"
    return text

# Function to upload audio file and display results
def upload_audio():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    
    recognized_text = recognize_speech(file_path)
    text_label.config(text=f"Recognized Text: {recognized_text}")

    predicted_emotion = predict_emotion(file_path)
    emotion_label.config(text=f"Detected Emotion: {predicted_emotion}")

root = tk.Tk()
root.geometry("600x400")
root.title("Speech Emotion Recognition")

upload_button = Button(root, text="Upload Audio", command=upload_audio, font=("Helvetica", 12), bg='blue', fg='white')
upload_button.pack(pady=20)

text_label = Label(root, text="Recognized Text: ", font=("Helvetica", 12), fg='black', wraplength=500)
text_label.pack(pady=10)

emotion_label = Label(root, text="Detected Emotion: ", font=("Helvetica", 12), fg='black', wraplength=500)
emotion_label.pack(pady=10)

root.mainloop()
