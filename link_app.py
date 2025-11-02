"""
Link App Module
Handles backend/frontend communication and data processing
"""

import os
import json
import base64
import cv2
import numpy as np
from datetime import datetime
from werkzeug.utils import secure_filename
from face_emotions import EmotionDetector


class EmotionAPIHandler:
    """
    Handler for emotion detection API operations
    """

    def __init__(self, upload_folder='datasets/uploads', model_path='models/emotion_model.h5'):
        """
        Initialize the API handler

        Args:
            upload_folder (str): Folder to save uploaded images
            model_path (str): Path to the emotion detection model
        """
        self.upload_folder = upload_folder
        self.detector = EmotionDetector(model_path)
        self.allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

        # Create upload folder if it doesn't exist
        os.makedirs(upload_folder, exist_ok=True)

    def allowed_file(self, filename):
        """
        Check if file extension is allowed

        Args:
            filename (str): Name of the file

        Returns:
            bool: True if allowed, False otherwise
        """
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.allowed_extensions

    def save_uploaded_file(self, file):
        """
        Save uploaded file to disk

        Args:
            file: Uploaded file object

        Returns:
            str: Path to saved file
        """
        if file and self.allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(self.upload_folder, filename)
            file.save(filepath)
            return filepath
        return None

    def process_image_file(self, file):
        """
        Process uploaded image file

        Args:
            file: Uploaded file object

        Returns:
            dict: Processing results
        """
        # Save the file
        filepath = self.save_uploaded_file(file)
        if not filepath:
            return {
                'success': False,
                'error': 'Invalid file type'
            }

        # Read and process image
        image = cv2.imread(filepath)
        if image is None:
            return {
                'success': False,
                'error': 'Could not read image'
            }

        # Detect emotions
        results = self.detector.detect_and_classify(image)

        # Draw results on image
        annotated_image = self.detector.draw_results(image, results)

        # Save annotated image
        output_filename = f"annotated_{os.path.basename(filepath)}"
        output_path = os.path.join(self.upload_folder, output_filename)
        cv2.imwrite(output_path, annotated_image)

        return {
            'success': True,
            'num_faces': len(results),
            'results': results,
            'original_image': filepath,
            'annotated_image': output_path
        }

    def process_base64_image(self, base64_string):
        """
        Process base64 encoded image (from webcam)

        Args:
            base64_string (str): Base64 encoded image

        Returns:
            dict: Processing results
        """
        try:
            # Remove header if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]

            # Decode base64 string
            image_data = base64.b64decode(base64_string)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                return {
                    'success': False,
                    'error': 'Could not decode image'
                }

            # Detect emotions
            results = self.detector.detect_and_classify(image)

            # Draw results on image
            annotated_image = self.detector.draw_results(image, results)

            # Save images
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            original_path = os.path.join(self.upload_folder, f"webcam_{timestamp}.jpg")
            annotated_path = os.path.join(self.upload_folder, f"webcam_annotated_{timestamp}.jpg")

            cv2.imwrite(original_path, image)
            cv2.imwrite(annotated_path, annotated_image)

            # Convert annotated image to base64
            _, buffer = cv2.imencode('.jpg', annotated_image)
            annotated_base64 = base64.b64encode(buffer).decode('utf-8')

            return {
                'success': True,
                'num_faces': len(results),
                'results': results,
                'original_image': original_path,
                'annotated_image': annotated_path,
                'annotated_base64': annotated_base64
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def save_labeled_data(self, image_path, emotion_label):
        """
        Save labeled data for future training

        Args:
            image_path (str): Path to the image
            emotion_label (str): Emotion label

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create emotion-specific folder in training data
            emotion_folder = os.path.join('datasets/training', emotion_label)
            os.makedirs(emotion_folder, exist_ok=True)

            # Read the original image
            image = cv2.imread(image_path)
            if image is None:
                return False

            # Detect face and crop
            faces = self.detector.detect_faces(image)
            if len(faces) == 0:
                return False

            # Take the first face
            x, y, w, h = faces[0]
            face = image[y:y+h, x:x+w]

            # Resize to standard size
            face_resized = cv2.resize(face, (48, 48))

            # Save to training folder
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{emotion_label}_{timestamp}.jpg"
            save_path = os.path.join(emotion_folder, filename)
            cv2.imwrite(save_path, face_resized)

            return True

        except Exception as e:
            print(f"Error saving labeled data: {e}")
            return False

    def get_statistics(self):
        """
        Get statistics about processed images

        Returns:
            dict: Statistics
        """
        stats = {
            'total_uploads': 0,
            'by_emotion': {}
        }

        # Count files in upload folder
        if os.path.exists(self.upload_folder):
            stats['total_uploads'] = len([
                f for f in os.listdir(self.upload_folder)
                if os.path.isfile(os.path.join(self.upload_folder, f))
            ])

        # Count training data by emotion
        training_dir = 'datasets/training'
        if os.path.exists(training_dir):
            for emotion in os.listdir(training_dir):
                emotion_path = os.path.join(training_dir, emotion)
                if os.path.isdir(emotion_path):
                    count = len([
                        f for f in os.listdir(emotion_path)
                        if os.path.isfile(os.path.join(emotion_path, f))
                    ])
                    stats['by_emotion'][emotion] = count

        return stats


def format_response(success, data=None, error=None):
    """
    Format API response

    Args:
        success (bool): Whether the operation was successful
        data (dict): Response data
        error (str): Error message if any

    Returns:
        dict: Formatted response
    """
    response = {
        'success': success,
        'timestamp': datetime.now().isoformat()
    }

    if data:
        response['data'] = data
    if error:
        response['error'] = error

    return response


def serialize_results(results):
    """
    Serialize detection results for JSON response

    Args:
        results (list): List of detection results

    Returns:
        list: Serialized results
    """
    serialized = []
    for result in results:
        serialized.append({
            'bbox': {
                'x': int(result['bbox'][0]),
                'y': int(result['bbox'][1]),
                'width': int(result['bbox'][2]),
                'height': int(result['bbox'][3])
            },
            'emotion': result['emotion'],
            'confidence': float(result['confidence'])
        })
    return serialized


if __name__ == "__main__":
    # Test the API handler
    handler = EmotionAPIHandler()

    print("API Handler initialized successfully")
    print(f"Upload folder: {handler.upload_folder}")
    print(f"Allowed extensions: {handler.allowed_extensions}")

    # Get statistics
    stats = handler.get_statistics()
    print(f"\nStatistics:")
    print(json.dumps(stats, indent=2))
