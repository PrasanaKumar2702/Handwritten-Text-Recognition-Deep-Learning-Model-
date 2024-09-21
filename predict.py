import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

char_list = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, (128, 32))

def preprocess_image(img):
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

def labels_to_text(labels):
    return ''.join([char_list[c] for c in labels])

def beam_search_decoder(predictions, top_k=3):
    # Placeholder for beam search decoder
    return "decoded_text"  # dummy return

def predict(img_path):
    img = preprocess_image(load_image(img_path))
    model = load_model('model/handwritten_text_recognition_model.h5')
    pred = model.predict(np.expand_dims(img, axis=0))
    decoded_text = beam_search_decoder(pred)
    return decoded_text

# Example usage
img_path = 'data/test/sample_image.png'
predicted_text = predict(img_path)
print("Predicted text:", predicted_text)
