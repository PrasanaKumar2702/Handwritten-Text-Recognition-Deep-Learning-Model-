
# Handwritten Text Recognition

This project implements a Handwritten Text Recognition (HTR) model using CNN, Bi-directional LSTM, and a Beam Search Decoder. The model is capable of recognizing text from images of handwritten documents.

## Setup

1. Install dependencies:

```bash
pip install numpy matplotlib tensorflow keras opencv-python
```

2. Download the IAM Handwriting Database [here](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database).

3. Update the paths to your dataset in the script files.

## Usage

### Train the Model

The model is trained using `train.py`. Make sure to place your training images and labels in the appropriate folder before running.

```bash
python scripts/train.py
```

### Predict Handwritten Text

Once the model is trained, you can predict the text from a new image using `predict.py`.

```bash
python scripts/predict.py
```

### Model Architecture

- **CNN**: Used for feature extraction.
- **BiLSTM**: Sequential modeling of the extracted features.
- **Beam Search Decoder**: Improves the accuracy of the predictions by considering multiple candidate sequences.

### Directory Structure

```
Handwritten-Text-Recognition/
│
├── model/
│   ├── handwritten_text_recognition_model.h5
├── data/
│   ├── train/
│   ├── test/
│       ├── sample_image.png  # Example test image
├── scripts/
│   ├── train.py
│   ├── predict.py
├── README.md
├── requirements.txt
```
