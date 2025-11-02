# Facial Emotion Recognition System

A comprehensive web-based application for detecting and classifying human emotions from facial images using deep learning.

## Features

- **Image Upload**: Upload photos to detect emotions from faces
- **Webcam Integration**: Real-time emotion detection using your webcam
- **Multiple Emotions**: Detects 7 emotions (angry, disgust, fear, happy, sad, surprise, neutral)
- **Face Detection**: Automatically detects multiple faces in images
- **Confidence Scores**: Provides confidence levels for each emotion prediction
- **Training System**: Includes model training capabilities with data augmentation
- **Statistics Dashboard**: Track uploads and training data distribution
- **Modern UI**: Clean, responsive web interface

## Project Structure

```
facial-emotion-recognition/
├── app.py                  # Main Flask web application
├── face_emotions.py        # Face detection and emotion classification module
├── model_training.py       # Model training script
├── link_app.py            # Backend API handler
├── requirements.txt        # Python dependencies
├── datasets/              # Dataset storage
│   ├── training/          # Training data (organized by emotion)
│   ├── validation/        # Validation data
│   └── uploads/           # Uploaded images
├── models/                # Trained model storage
├── static/                # Static web assets
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
└── templates/             # HTML templates
    └── index.html
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Webcam (optional, for real-time detection)

### Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd facial-emotion-recognition
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download or train a model**

   Option 1: Use a pre-trained model (if available)
   - Place the model file in the `models/` directory as `emotion_model.h5`

   Option 2: Train your own model (see Training section below)

## Usage

### Running the Web Application

1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Use the application:
   - **Upload Tab**: Drag & drop or select an image file
   - **Webcam Tab**: Start camera and capture images for analysis
   - **Statistics Tab**: View processing statistics

### Command Line Usage

You can also use the face detection module directly from the command line:

```bash
python face_emotions.py path/to/image.jpg output.jpg
```

## Training the Model

### Using FER2013 Dataset

1. **Download the FER2013 dataset**
   - Available on [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
   - Download and extract to get `fer2013.csv`

2. **Prepare the dataset**
   ```bash
   python model_training.py --prepare-fer2013 path/to/fer2013.csv
   ```

3. **Train the model**
   ```bash
   python model_training.py --epochs 50 --batch-size 32
   ```

### Using Custom Dataset

1. **Organize your data**
   ```
   datasets/
   ├── training/
   │   ├── angry/
   │   ├── disgust/
   │   ├── fear/
   │   ├── happy/
   │   ├── sad/
   │   ├── surprise/
   │   └── neutral/
   └── validation/
       ├── angry/
       ├── disgust/
       ├── fear/
       ├── happy/
       ├── sad/
       ├── surprise/
       └── neutral/
   ```

2. **Train the model**
   ```bash
   python model_training.py
   ```

### Training Options

```bash
python model_training.py --help
```

Available options:
- `--train-dir`: Training data directory (default: datasets/training)
- `--val-dir`: Validation data directory (default: datasets/validation)
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 32)
- `--model-path`: Path to save trained model (default: models/emotion_model.h5)

## Model Architecture

The emotion detection model uses a Convolutional Neural Network (CNN) with the following architecture:

- 4 Convolutional blocks with BatchNormalization and Dropout
- Progressive filter sizes: 32 → 64 → 128 → 256
- MaxPooling for downsampling
- 2 Fully connected layers (512 and 256 units)
- Softmax output layer for 7 emotion classes
- Total parameters: ~8M

### Data Augmentation

Training uses real-time data augmentation:
- Random rotation (±20°)
- Width/height shifts (±20%)
- Horizontal flips
- Zoom (±20%)
- Shear transformations

## API Endpoints

The application provides RESTful API endpoints:

### POST /upload
Upload and analyze an image file

**Request**: `multipart/form-data` with `file` field

**Response**:
```json
{
  "success": true,
  "data": {
    "num_faces": 1,
    "results": [
      {
        "bbox": {"x": 100, "y": 100, "width": 150, "height": 150},
        "emotion": "happy",
        "confidence": 0.95
      }
    ],
    "annotated_image": "/uploads/filename.jpg"
  }
}
```

### POST /webcam
Analyze base64-encoded webcam image

**Request**:
```json
{
  "image": "base64_encoded_image_data"
}
```

### GET /statistics
Get processing statistics

**Response**:
```json
{
  "success": true,
  "data": {
    "total_uploads": 42,
    "by_emotion": {
      "happy": 150,
      "sad": 120,
      "angry": 80
    }
  }
}
```

### GET /health
Health check endpoint

## Technologies Used

- **Backend**: Flask (Python web framework)
- **Machine Learning**: TensorFlow/Keras
- **Computer Vision**: OpenCV
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Data Processing**: NumPy, Pandas, scikit-learn

## Performance

- **Inference Speed**: ~50-100ms per image (CPU)
- **Model Size**: ~30MB
- **Input Size**: 48x48 grayscale images
- **Accuracy**: ~65-70% on FER2013 test set (varies with training)

## Troubleshooting

### Model not found error
- Make sure you have trained a model or placed a pre-trained model in `models/emotion_model.h5`

### Webcam not working
- Ensure your browser has camera permissions
- Check if another application is using the camera
- Try a different browser (Chrome/Firefox recommended)

### Memory errors during training
- Reduce batch size: `--batch-size 16`
- Use fewer training images
- Close other memory-intensive applications

### Low accuracy
- Train for more epochs
- Use more training data
- Adjust learning rate
- Try different data augmentation parameters

## Future Enhancements

- [ ] Real-time video emotion detection
- [ ] Export predictions to CSV
- [ ] Model comparison dashboard
- [ ] User authentication
- [ ] Cloud deployment guide
- [ ] Mobile application
- [ ] Multi-language support

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- FER2013 dataset creators
- OpenCV community
- TensorFlow/Keras developers

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainer.

---

**Built with Python, TensorFlow, and Flask**
