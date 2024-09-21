import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, LSTM, Bidirectional
from tensorflow.keras.models import Model
import tensorflow as tf

char_list = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, (128, 32))

def preprocess_image(img):
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

def text_to_labels(text):
    return [char_list.index(c) for c in text]

def labels_to_text(labels):
    return ''.join([char_list[c] for c in labels])

input_shape = (32, 128, 1)
num_classes = len(char_list) + 1

def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Reshape((-1, x.shape[-1]))(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

model = build_model(input_shape, num_classes)

def ctc_loss(y_true, y_pred):
    input_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
    label_length = tf.fill([tf.shape(y_true)[0]], tf.shape(y_true)[1])
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

model.compile(optimizer='adam', loss=ctc_loss)

# Assuming train_images and train_labels are available
# Preprocess images and labels
train_images = np.array([preprocess_image(img) for img in train_images])
train_labels = np.array([text_to_labels(text) for text in train_labels])
train_labels = pad_sequences(train_labels, maxlen=max_seq_len, padding='post')

model.fit(train_images, train_labels, batch_size=32, epochs=100)
model.save('model/handwritten_text_recognition_model.h5')
