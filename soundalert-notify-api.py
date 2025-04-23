from flask import Flask, request, jsonify
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import librosa
import io
import csv

app = Flask(__name__)

yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

with open('yamnet_class_map.csv') as csv_file:
    reader = csv.reader(csv_file)
    next(reader)
    class_names = [row[2] for row in reader]

target_sounds = {
    'Fire alarm': '🚨 Fire Alarm',
    'Smoke detector, smoke alarm': '🚨 Fire Alarm',
    'Alarm clock': '🚨 Fire Alarm',
    'Doorbell': '🔔 Doorbell',
    'Vehicle horn, car horn, honking': '🚗 Car Horn',
    'Vehicle': '🚗 Car Horn'
}

@app.route('/classify', methods=['POST'])
def classify_audio():
    audio_file = request.files['audio'].read()
    audio_data, sr = librosa.load(io.BytesIO(audio_file), sr=16000, mono=True)
    scores, _, _ = yamnet_model(audio_data)
    mean_scores = np.mean(scores.numpy(), axis=0)
    top_class = class_names[mean_scores.argmax()]
    confidence = mean_scores.max()

    alert = target_sounds.get(top_class, "Other sound")
    
    return jsonify({
        'detected_class': top_class,
        'alert': alert,
        'confidence': float(confidence)
    })

if __name__ == '__main__':
    app.run(debug=True)
