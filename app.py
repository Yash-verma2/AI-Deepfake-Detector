import os
import numpy as np
import librosa
import joblib
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)
model = joblib.load("model/fakevoice_model.pkl")
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/team")
def team():
    return render_template("team.html")



@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["audio"]
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Extract features
        audio, sr = librosa.load(filepath, duration=7)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mean_mfcc = np.mean(mfcc.T, axis=0).reshape(1, -1)

        # Predict
        prediction = model.predict(mean_mfcc)[0]
        
        # Get model confidence (probability score)
        try:
            probabilities = model.predict_proba(mean_mfcc)[0]
            confidence = float(round(np.max(probabilities) * 100, 2))
        except AttributeError:
            confidence = 100.0  # Fallback if model has no predict_proba

        verdict = "FAKE" if prediction == 1 else "REAL"

        # Handle AJAX/JS request
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"prediction": verdict, "confidence": confidence})

        # Fallback for HTML rendering
        result_text = "Fake Voice (AI Generated)" if prediction == 1 else "Real Voice"
        return render_template("index.html", result=result_text)

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
