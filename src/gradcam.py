import numpy as np
import tensorflow as tf
import cv2
from src.config import IMAGE_SIZE
from src.predictor import preprocess_image


def find_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    for layer in reversed(model.layers):
        if hasattr(layer, "layers"):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return sub_layer.name
    return None


def make_gradcam_heatmap(model, image, last_conv_layer_name=None):
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer_name(model)
    if last_conv_layer_name is None:
        return None, "No convolution layer found for Grad-CAM."
    img_array = preprocess_image(image)
    try:
        conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = tf.keras.models.Model([model.inputs], [conv_layer.output, model.output])
    except Exception:
        return None, "Grad-CAM is not available for the current model structure."
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        return None, "Could not calculate gradients for Grad-CAM."
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0)
    max_value = np.max(heatmap)
    if max_value == 0:
        return None, "Grad-CAM heatmap has no activated region."
    heatmap /= max_value
    return heatmap.numpy(), None


def overlay_heatmap(image, heatmap, alpha: float = 0.4):
    image = image.convert("RGB").resize(IMAGE_SIZE)
    img = np.array(image)
    heatmap_resized = cv2.resize(heatmap, IMAGE_SIZE)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img, 1 - alpha, colored, alpha, 0)
