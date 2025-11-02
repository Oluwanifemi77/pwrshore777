"""
Face Emotion Detection Module
Handles face detection and emotion classification
"""

import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
import os

class EmotionDetector:
    """
    Class for detecting faces and classifying emotions
    """

    # Emotion labels that the model can predict
    EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    def __init__(self, model_path='models/emotion_model.h5'):
        """
        Initialize the emotion detector

        Args:
            model_path (str): Path to the trained emotion detection model
        """
        self.model_path = model_path
        self.model = None

        # Load the face detection classifier (Haar Cascade)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # Load the emotion detection model if it exists
        if os.path.exists(model_path):
            self.load_model()
        else:
            print(f"Model not found at {model_path}. Please train the model first.")

    def load_model(self):
        """
        Load the trained emotion detection model
        """
        try:
            self.model = load_model(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def detect_faces(self, image):
        """
        Detect faces in an image

        Args:
            image: Input image (numpy array)

        Returns:
            List of face coordinates (x, y, w, h)
        """
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        return faces

    def preprocess_face(self, face_image):
        """
        Preprocess face image for emotion detection

        Args:
            face_image: Cropped face image

        Returns:
            Preprocessed image ready for model prediction
        """
        # Convert to grayscale
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

        # Resize to 48x48 (standard size for emotion detection models)
        resized = cv2.resize(gray, (48, 48))

        # Normalize pixel values
        normalized = resized / 255.0

        # Reshape for model input (batch_size, height, width, channels)
        reshaped = normalized.reshape(1, 48, 48, 1)

        return reshaped

    def predict_emotion(self, face_image):
        """
        Predict emotion from a face image

        Args:
            face_image: Cropped face image

        Returns:
            Tuple of (emotion_label, confidence)
        """
        if self.model is None:
            return "unknown", 0.0

        # Preprocess the face
        processed_face = self.preprocess_face(face_image)

        # Make prediction
        predictions = self.model.predict(processed_face, verbose=0)

        # Get the emotion with highest probability
        emotion_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][emotion_idx])
        emotion = self.EMOTIONS[emotion_idx]

        return emotion, confidence

    def detect_and_classify(self, image):
        """
        Detect faces and classify emotions in an image

        Args:
            image: Input image

        Returns:
            List of dictionaries containing face coordinates, emotion, and confidence
        """
        results = []

        # Detect faces
        faces = self.detect_faces(image)

        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = image[y:y+h, x:x+w]

            # Predict emotion
            emotion, confidence = self.predict_emotion(face_roi)

            # Store results
            results.append({
                'bbox': (x, y, w, h),
                'emotion': emotion,
                'confidence': confidence
            })

        return results

    def draw_results(self, image, results):
        """
        Draw bounding boxes and emotion labels on the image

        Args:
            image: Input image
            results: List of detection results

        Returns:
            Image with annotations
        """
        output_image = image.copy()

        for result in results:
            x, y, w, h = result['bbox']
            emotion = result['emotion']
            confidence = result['confidence']

            # Draw rectangle around face
            cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Create label text
            label = f"{emotion}: {confidence:.2f}"

            # Draw label background
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                output_image,
                (x, y - label_height - 10),
                (x + label_width, y),
                (0, 255, 0),
                -1
            )

            # Draw label text
            cv2.putText(
                output_image,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return output_image


def process_image(image_path, output_path=None):
    """
    Process an image file and detect emotions

    Args:
        image_path (str): Path to input image
        output_path (str): Optional path to save annotated image

    Returns:
        List of detection results
    """
    # Initialize detector
    detector = EmotionDetector()

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return []

    # Detect and classify
    results = detector.detect_and_classify(image)

    # Draw results if output path is provided
    if output_path:
        annotated_image = detector.draw_results(image, results)
        cv2.imwrite(output_path, annotated_image)
        print(f"Annotated image saved to {output_path}")

    return results


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "output.jpg"

        results = process_image(image_path, output_path)

        print(f"\nDetected {len(results)} face(s):")
        for i, result in enumerate(results, 1):
            print(f"Face {i}: {result['emotion']} (confidence: {result['confidence']:.2f})")
    else:
        print("Usage: python face_emotions.py <image_path> [output_path]")
