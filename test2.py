import cv2
import numpy as np
import tensorflow as tf

def display_emotion(frame, model):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (0, 0, 255)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Load Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi_color = frame[y:y+h, x:x+w]
        face_roi_rgb = cv2.cvtColor(face_roi_color, cv2.COLOR_BGR2RGB)
        try:
            resized_image = cv2.resize(face_roi_rgb, (224, 224))  # Adjust based on model input
        except:
            continue  # Skip if resizing fails
        
        normalized = resized_image / 255.0  # Normalize
        final_image = np.expand_dims(normalized, axis=0)  # (1, 224, 224, 3)

        predictions = model.predict(final_image, verbose=0)
        predicted_label = class_labels[np.argmax(predictions)]

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), text_color, 2)
        cv2.putText(frame, predicted_label, (x, y - 10), font, 0.8, text_color, 2)

    return frame

def main():
    model = tf.keras.models.load_model('Emotion-detection/emotion_detection.keras')

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = display_emotion(frame, model)
        cv2.imshow('Facial Expression Recognition', frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
