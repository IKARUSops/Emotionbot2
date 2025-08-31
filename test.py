import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Make sure you're using the correct backend
os.environ["KERAS_BACKEND"] = "tensorflow"  # use TensorFlow for loading .keras or .h5

# Load the model (local copy of Hugging Face one)
model = load_model("Emotion-detection/emotion_detection.keras")

# Emotion labels used in the Hugging Face model
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Adjust size according to the model
IMG_SIZE = 224  # The HF model expects 224x224 RGB input

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB (if model was trained on color)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_face = cv2.resize(rgb_frame, (IMG_SIZE, IMG_SIZE))
    normalized_face = resized_face / 255.0

    # Prepare input shape (1, 224, 224, 3)
    input_tensor = np.expand_dims(normalized_face, axis=0)

    # Predict emotion
    predictions = model.predict(input_tensor, verbose=0)
    predicted_emotion = emotion_labels[np.argmax(predictions)]

    # Show result
    cv2.putText(frame, predicted_emotion, (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
