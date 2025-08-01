import os
os.environ['NUMBA_DISABLE_CACHE'] = '1'  # ✅ Must come first!

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import librosa
import numpy as np
import uuid

app = Flask(__name__)
CORS(app)

# Load your trained model
model = joblib.load('model/fakevoice_model.pkl')

UPLOAD_FOLDER = '/tmp/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_features(audio_path):
    audio, sr = librosa.load(audio_path, duration=3)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mean_mfcc = np.mean(mfcc.T, axis=0)
    return mean_mfcc

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    filename = str(uuid.uuid4()) + ".wav"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    features = extract_features(file_path)
    if features is None:
        return jsonify({'error': 'Feature extraction failed'}), 500

    features = features.reshape(1, -1)  # 💡 Reshape is required
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]

    label = 'Fake' if prediction == 1 else 'Real'
    confidence = round(float(np.max(proba)) * 100, 2)

    os.remove(file_path)

    return jsonify({
        'prediction': label,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=7860)
