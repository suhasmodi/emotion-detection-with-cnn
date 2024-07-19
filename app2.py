import os
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
import sounddevice as sd
import speech_recognition as sr
import tkinter as tk
from tkinter import filedialog, messagebox
from threading import Thread

# Load the trained emotion detection model
model = tf.keras.models.load_model('SER_cnn_model.h5')

# Emotion labels
emotion_list = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

class EmotionDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Audio Emotion Detector")
        self.root.geometry("400x400")

        self.label = tk.Label(root, text="Choose an option below")
        self.label.pack(pady=20)

        self.record_button = tk.Button(root, text="Start Live Recording", command=self.start_recording)
        self.record_button.pack(pady=10)

        self.stop_button = tk.Button(root, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(pady=10)

        self.upload_button = tk.Button(root, text="Upload Audio File", command=self.upload_audio_file)
        self.upload_button.pack(pady=10)

        self.clear_button = tk.Button(root, text="Clear", command=self.clear_result)
        self.clear_button.pack(pady=10)

        self.result_label = tk.Label(root, text="")
        self.result_label.pack(pady=20)

        self.audio_data = np.array([])
        self.recording_stream = None

    def start_recording(self):
        self.result_label.config(text="Recording...")
        self.disable_buttons()
        self.audio_data = np.array([])
        self.recording_stream = sd.InputStream(callback=self.audio_callback)
        self.recording_stream.start()
        self.stop_button.config(state=tk.NORMAL)

    def stop_recording(self):
        if self.recording_stream is not None:
            self.recording_stream.stop()
            self.recording_stream.close()
            self.recording_stream = None
            text, emotion = self.process_audio_data(self.audio_data)
            self.result_label.config(text=f"Detected Text: {text}\nDetected Emotion: {emotion}")
            self.enable_buttons()

    def upload_audio_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav;*.mp3;*.ogg")])
        if file_path:
            self.result_label.config(text="Processing...")
            self.disable_buttons()
            Thread(target=self.process_uploaded_audio, args=(file_path,)).start()

    def process_uploaded_audio(self, file_path):
        try:
            audio_data, _ = librosa.load(file_path, sr=22050)
            text, emotion = self.process_audio_data(audio_data)
            self.result_label.config(text=f"Detected Text: {text}\nDetected Emotion: {emotion}")
            print(text)
        except Exception as e:
            messagebox.showerror("Error", f"Error processing uploaded audio: {str(e)}")
        finally:
            self.enable_buttons()

    def clear_result(self):
        self.result_label.config(text="")

    def disable_buttons(self):
        self.record_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        self.upload_button.config(state=tk.DISABLED)
        self.clear_button.config(state=tk.DISABLED)

    def enable_buttons(self):
        self.record_button.config(state=tk.NORMAL)
        self.upload_button.config(state=tk.NORMAL)
        self.clear_button.config(state=tk.NORMAL)

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status, flush=True)
        self.audio_data = np.concatenate([self.audio_data, indata.flatten()])

    def process_audio_data(self, audio_data):
        # Transcribe audio to text
        temp_audio_path = "temp_audio.wav"
        sf.write(temp_audio_path, audio_data, 22050)

        recognizer = sr.Recognizer()
        audio_file = sr.AudioFile(temp_audio_path)
        
        with audio_file as source:
            audio = recognizer.record(source)
        
        try:
            text = recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            text = "Could not understand audio"
        except sr.RequestError as e:
            text = f"Could not request results from Google Speech Recognition service; {e}"
        
        # Predict emotion
        emotion = self.predict_emotion(audio_data)

        os.remove(temp_audio_path)  # Clean up temporary file
        return text, emotion

    def predict_emotion(self, audio_data):
        features = self.extract_features(audio_data)
        features = np.expand_dims(features, axis=0)
        features = features[..., np.newaxis]
        prediction = model.predict(features)
        emotion = emotion_list[np.argmax(prediction)]
        return emotion

    def extract_features(self, audio, sr=22050):
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetectorApp(root)
    root.mainloop()
