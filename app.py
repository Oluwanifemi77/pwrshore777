"""
Main Web Application
Flask application for facial emotion recognition
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
from link_app import EmotionAPIHandler, format_response, serialize_results

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'datasets/uploads'
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'

# Initialize API handler
api_handler = EmotionAPIHandler()


@app.route('/')
def index():
    """
    Home page
    """
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    """
    Handle image upload and process for emotion detection
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify(format_response(
                success=False,
                error='No file provided'
            )), 400

        file = request.files['file']

        # Check if file is selected
        if file.filename == '':
            return jsonify(format_response(
                success=False,
                error='No file selected'
            )), 400

        # Process the image
        result = api_handler.process_image_file(file)

        if not result['success']:
            return jsonify(format_response(
                success=False,
                error=result.get('error', 'Processing failed')
            )), 400

        # Serialize results
        serialized_results = serialize_results(result['results'])

        return jsonify(format_response(
            success=True,
            data={
                'num_faces': result['num_faces'],
                'results': serialized_results,
                'annotated_image': f"/uploads/{os.path.basename(result['annotated_image'])}"
            }
        ))

    except Exception as e:
        return jsonify(format_response(
            success=False,
            error=str(e)
        )), 500


@app.route('/webcam', methods=['POST'])
def process_webcam():
    """
    Process webcam image (base64 encoded)
    """
    try:
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify(format_response(
                success=False,
                error='No image data provided'
            )), 400

        # Process the base64 image
        result = api_handler.process_base64_image(data['image'])

        if not result['success']:
            return jsonify(format_response(
                success=False,
                error=result.get('error', 'Processing failed')
            )), 400

        # Serialize results
        serialized_results = serialize_results(result['results'])

        return jsonify(format_response(
            success=True,
            data={
                'num_faces': result['num_faces'],
                'results': serialized_results,
                'annotated_image': f"data:image/jpeg;base64,{result['annotated_base64']}"
            }
        ))

    except Exception as e:
        return jsonify(format_response(
            success=False,
            error=str(e)
        )), 500


@app.route('/label', methods=['POST'])
def label_image():
    """
    Save labeled image for training
    """
    try:
        data = request.get_json()

        if not data or 'image_path' not in data or 'emotion' not in data:
            return jsonify(format_response(
                success=False,
                error='Missing required fields'
            )), 400

        # Save labeled data
        success = api_handler.save_labeled_data(
            data['image_path'],
            data['emotion']
        )

        if success:
            return jsonify(format_response(
                success=True,
                data={'message': 'Image labeled and saved successfully'}
            ))
        else:
            return jsonify(format_response(
                success=False,
                error='Failed to save labeled image'
            )), 400

    except Exception as e:
        return jsonify(format_response(
            success=False,
            error=str(e)
        )), 500


@app.route('/statistics')
def get_statistics():
    """
    Get statistics about processed images
    """
    try:
        stats = api_handler.get_statistics()
        return jsonify(format_response(
            success=True,
            data=stats
        ))
    except Exception as e:
        return jsonify(format_response(
            success=False,
            error=str(e)
        )), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    Serve uploaded files
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/health')
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': api_handler.detector.model is not None
    })


@app.errorhandler(413)
def too_large(e):
    """
    Handle file too large error
    """
    return jsonify(format_response(
        success=False,
        error='File too large. Maximum size is 16MB.'
    )), 413


@app.errorhandler(404)
def not_found(e):
    """
    Handle 404 errors
    """
    return jsonify(format_response(
        success=False,
        error='Resource not found'
    )), 404


@app.errorhandler(500)
def internal_error(e):
    """
    Handle internal server errors
    """
    return jsonify(format_response(
        success=False,
        error='Internal server error'
    )), 500


if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('datasets/uploads', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Run the app
    print("Starting Facial Emotion Recognition App...")
    print("Open your browser and navigate to http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
