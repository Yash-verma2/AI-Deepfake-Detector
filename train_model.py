import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Sample dataset path (make folders real/ and fake/ with some .wav files)
DATASET = {
    "real": "dataset/real/",
    "fake": "dataset/fake/"
}

X, y = [], []

for label in DATASET:
    folder = DATASET[label]
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            audio, sr = librosa.load(os.path.join(folder, file), duration=3)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mean_mfcc = np.mean(mfcc.T, axis=0)
            X.append(mean_mfcc)
            y.append(0 if label == "real" else 1)

clf = RandomForestClassifier()
clf.fit(X, y)

# Save model
joblib.dump(clf, "model/fakevoice_model.pkl")
print("Model saved.")
