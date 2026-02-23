from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
from risk_analysis import analyze_risk

# Initialize Flask app
app = Flask(__name__)

# Load trained model ONCE
model = tf.keras.models.load_model("crop_disease_model.h5")

# Class labels (same order as training)
classes = ["early_blight", "healthy", "late_blight", "leaf_mold", "septoria"]

# Image preprocessing function
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ---------------- HOME / UI ROUTE ----------------
from flask import Flask, request, jsonify, render_template
# (keep the rest of your imports as-is)

@app.route("/")
def home():
    return render_template("index.html")




# ---------------- UI PREDICTION ROUTE ----------------
@app.route("/predict-ui", methods=["POST"])
def predict_ui():
    if "image" not in request.files:
        return render_template("index.html", error="No image uploaded")

    file = request.files["image"]
    img = Image.open(file).convert("RGB")

    processed_img = preprocess_image(img)
    preds = model.predict(processed_img)[0]
    idx = np.argmax(preds)

    disease = classes[idx]
    confidence = round(float(preds[idx]) * 100, 2)
    risk = analyze_risk(disease, preds[idx])

    result = {
        "disease": disease,
        "confidence": confidence,
        "risk_analysis": risk
    }

    return render_template("index.html", result=result)


# ---------------- API ROUTE (JSON) ----------------
@app.route("/predict", methods=["POST"])
def predict_api():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img = Image.open(file).convert("RGB")

    processed_img = preprocess_image(img)
    preds = model.predict(processed_img)[0]
    idx = np.argmax(preds)

    disease = classes[idx]
    confidence = float(preds[idx])
    risk = analyze_risk(disease, confidence)

    return jsonify({
        "disease": disease,
        "confidence": round(confidence * 100, 2),
        "risk_analysis": risk
    })

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)