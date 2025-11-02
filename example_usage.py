"""
Example Usage of Facial Emotion Recognition System
Demonstrates how to use the emotion detection module programmatically
"""

import cv2
import numpy as np
from face_emotions import EmotionDetector


def example_1_detect_from_file():
    """Example 1: Detect emotions from an image file"""
    print("="*60)
    print("Example 1: Detect Emotions from Image File")
    print("="*60)

    # Initialize detector
    detector = EmotionDetector()

    # Path to your image
    image_path = "path/to/your/image.jpg"  # Change this to your image path

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        print(f"ERROR: Could not read image from {image_path}")
        return

    # Detect faces and classify emotions
    results = detector.detect_and_classify(image)

    # Print results
    print(f"\nDetected {len(results)} face(s):\n")
    for i, result in enumerate(results, 1):
        x, y, w, h = result['bbox']
        emotion = result['emotion']
        confidence = result['confidence']

        print(f"Face {i}:")
        print(f"  Position: ({x}, {y}), Size: {w}x{h}")
        print(f"  Emotion: {emotion}")
        print(f"  Confidence: {confidence:.2%}")
        print()

    # Draw results on image
    annotated_image = detector.draw_results(image, results)

    # Save annotated image
    output_path = "output_annotated.jpg"
    cv2.imwrite(output_path, annotated_image)
    print(f"Annotated image saved to: {output_path}")


def example_2_detect_from_webcam():
    """Example 2: Real-time emotion detection from webcam"""
    print("="*60)
    print("Example 2: Real-time Webcam Emotion Detection")
    print("="*60)
    print("Press 'q' to quit\n")

    # Initialize detector
    detector = EmotionDetector()

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not access webcam")
        return

    try:
        while True:
            # Read frame
            ret, frame = cap.read()

            if not ret:
                print("ERROR: Failed to read frame")
                break

            # Detect emotions
            results = detector.detect_and_classify(frame)

            # Draw results
            annotated_frame = detector.draw_results(frame, results)

            # Display
            cv2.imshow('Emotion Detection', annotated_frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


def example_3_batch_processing():
    """Example 3: Batch process multiple images"""
    print("="*60)
    print("Example 3: Batch Process Multiple Images")
    print("="*60)

    # Initialize detector
    detector = EmotionDetector()

    # List of image paths
    image_paths = [
        "image1.jpg",
        "image2.jpg",
        "image3.jpg"
    ]  # Change these to your image paths

    # Process each image
    all_results = []

    for i, image_path in enumerate(image_paths, 1):
        print(f"\nProcessing image {i}/{len(image_paths)}: {image_path}")

        # Read image
        image = cv2.imread(image_path)

        if image is None:
            print(f"  WARNING: Could not read {image_path}")
            continue

        # Detect emotions
        results = detector.detect_and_classify(image)

        # Store results
        all_results.append({
            'image': image_path,
            'num_faces': len(results),
            'results': results
        })

        # Print summary
        print(f"  Found {len(results)} face(s)")
        for result in results:
            print(f"    - {result['emotion']} ({result['confidence']:.2%})")

    # Print overall summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total images processed: {len(all_results)}")
    print(f"Total faces detected: {sum(r['num_faces'] for r in all_results)}")

    # Count emotions
    emotion_counts = {}
    for result in all_results:
        for face_result in result['results']:
            emotion = face_result['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    print("\nEmotion distribution:")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {emotion}: {count}")


def example_4_custom_processing():
    """Example 4: Custom processing with emotion predictions"""
    print("="*60)
    print("Example 4: Custom Processing with Predictions")
    print("="*60)

    # Initialize detector
    detector = EmotionDetector()

    # Read image
    image_path = "path/to/your/image.jpg"  # Change this
    image = cv2.imread(image_path)

    if image is None:
        print(f"ERROR: Could not read image from {image_path}")
        return

    # Detect faces
    faces = detector.detect_faces(image)

    print(f"Found {len(faces)} face(s)\n")

    # Process each face individually
    for i, (x, y, w, h) in enumerate(faces, 1):
        # Extract face
        face_roi = image[y:y+h, x:x+w]

        # Predict emotion
        emotion, confidence = detector.predict_emotion(face_roi)

        print(f"Face {i}:")
        print(f"  Bounding box: ({x}, {y}, {w}, {h})")
        print(f"  Predicted emotion: {emotion}")
        print(f"  Confidence: {confidence:.2%}")

        # Custom logic based on emotion
        if emotion == 'happy' and confidence > 0.8:
            print("  -> High confidence happiness detected!")
        elif emotion in ['sad', 'angry'] and confidence > 0.7:
            print("  -> Negative emotion detected")

        print()


def main():
    """Main function to run examples"""
    print("\n" + "="*60)
    print("FACIAL EMOTION RECOGNITION - USAGE EXAMPLES")
    print("="*60)
    print("\nChoose an example to run:")
    print("1. Detect emotions from image file")
    print("2. Real-time webcam detection")
    print("3. Batch process multiple images")
    print("4. Custom processing example")
    print("0. Exit")

    while True:
        choice = input("\nEnter your choice (0-4): ").strip()

        if choice == '0':
            print("Exiting...")
            break
        elif choice == '1':
            example_1_detect_from_file()
        elif choice == '2':
            example_2_detect_from_webcam()
        elif choice == '3':
            example_3_batch_processing()
        elif choice == '4':
            example_4_custom_processing()
        else:
            print("Invalid choice. Please enter 0-4.")

        print("\n" + "-"*60)
        input("Press Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
