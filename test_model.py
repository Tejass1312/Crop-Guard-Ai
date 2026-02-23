print("🔥 test_model.py is running 🔥")

import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("crop_disease_model.h5")

# Class labels (MUST match training order)
classes = ["early_blight", "healthy", "late_blight", "leaf_mold", "septoria"]

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    idx = np.argmax(preds)

    disease = classes[idx]
    confidence = preds[idx] * 100

    print("Predicted Disease:", disease)
    print("Confidence:", round(confidence, 2), "%")

# 🔴 THIS WAS MISSING
predict_image("test_leaf.jpg")
