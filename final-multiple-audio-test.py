import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import csv
import os

# Load YAMNet model
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Load class names correctly
with open('yamnet_class_map.csv') as csv_file:
    reader = csv.reader(csv_file)
    next(reader)
    class_names = [row[2] for row in reader]

# Your target sound mappings
target_sounds = {
    'Fire alarm': '🚨 Fire Alarm',
    'Smoke detector, smoke alarm': '🚨 Fire Alarm',
    'Alarm clock': '🚨 Fire Alarm',  # Consider adding if relevant
    'Doorbell': '🔔 Doorbell',
    'Ding-dong': '🔔 Doorbell',      # Add to handle vintage/classic doorbells
    'Vehicle horn, car horn, honking': '🚗 Car Horn',
    'Vehicle': '🚗 Car Horn',         # Add generic "Vehicle" to strengthen detection
    'Car alarm': '🚗 Car Horn',       # Optional, could help accuracy
}

# Function to classify audio file
def classify_audio(audio_path):
    audio_data, sr = librosa.load(audio_path, sr=16000, mono=True)
    scores, _, _ = yamnet_model(audio_data)
    mean_scores = np.mean(scores.numpy(), axis=0)
    top_class = class_names[mean_scores.argmax()]
    confidence = mean_scores.max()

    # This is exactly where you update the confidence threshold
    if top_class in target_sounds and confidence > 0.2:
        alert = target_sounds[top_class]
    else:
        alert = "❌ Other sound"

    print(f"{audio_path}: Detected {alert} ({top_class}), Confidence: {confidence:.2f}")

# Automatically test all MP3 files in current folder
audio_files = [f for f in os.listdir('.') if f.lower().endswith('.mp3')]

for audio_file in audio_files:
    classify_audio(audio_file)
