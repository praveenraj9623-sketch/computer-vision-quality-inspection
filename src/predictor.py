from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from src.config import IMAGE_SIZE, CLASS_DISPLAY_NAMES


def load_model_safe(model_path: str):
    path = Path(model_path)
    if not path.exists():
        return None, f"Model file not found: {path}"
    try:
        model = tf.keras.models.load_model(path)
        return model, None
    except Exception as error:
        return None, str(error)


def preprocess_image(image: Image.Image):
    image = image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.array(image).astype("float32")
    return np.expand_dims(arr, axis=0)


def predict_image(model, image: Image.Image, threshold: float = 0.5):
    arr = preprocess_image(image)
    probability_ok = float(model.predict(arr, verbose=0)[0][0])
    if probability_ok >= threshold:
        predicted_class = "ok_front"
        confidence = probability_ok
    else:
        predicted_class = "def_front"
        confidence = 1 - probability_ok
    return {
        "predicted_class": predicted_class,
        "prediction_label": CLASS_DISPLAY_NAMES[predicted_class],
        "confidence": float(confidence),
        "probability_ok": probability_ok,
        "probability_defective": float(1 - probability_ok),
    }
