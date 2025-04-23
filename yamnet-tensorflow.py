import csv
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa

# Load YAMNet model from TensorFlow Hub
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

# Load class labels correctly
class_map_path = 'yamnet_class_map.csv'

with open(class_map_path) as csv_file:
    reader = csv.reader(csv_file)
    next(reader)  # skip header
    class_names = [row[2] for row in reader]  # use descriptive names

# Function to classify audio file
def classify_audio(audio_path):
    audio_data, sr = librosa.load(audio_path, sr=16000, mono=True)
    scores, embeddings, spectrogram = yamnet_model(audio_data)
    mean_scores = np.mean(scores.numpy(), axis=0)
    top_class = class_names[mean_scores.argmax()]
    confidence = mean_scores.max()
    return top_class, confidence

# Test classification and print results
audio_path = 'firealarmsound.mp3'  # Update with your test audio filename
predicted_class, confidence = classify_audio(audio_path)
print(f"Predicted sound: {predicted_class}, Confidence: {confidence:.2f}")
