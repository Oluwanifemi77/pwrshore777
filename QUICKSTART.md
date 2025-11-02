# Quick Start Guide

Get up and running with the Facial Emotion Recognition system in 5 minutes!

## Prerequisites

- Python 3.8 or higher installed
- A webcam (optional, for real-time detection)

## Step 1: Installation

```bash
# Clone the repository (if not already done)
git clone <your-repo-url>
cd facial-emotion-recognition

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Download Pre-trained Model (Option 1 - Recommended for Quick Start)

For a quick start, you can download a pre-trained emotion detection model:

1. Download a pre-trained FER2013 model from:
   - [Kaggle Models](https://www.kaggle.com/models)
   - Or train your own (see Option 2)

2. Place the model file in the `models/` directory:
   ```
   models/emotion_model.h5
   ```

## Step 2: Train Your Own Model (Option 2 - Advanced)

If you want to train your own model:

1. **Download FER2013 dataset**:
   - Visit [Kaggle FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
   - Download and extract `fer2013.csv`

2. **Prepare and train**:
   ```bash
   # Prepare dataset
   python model_training.py --prepare-fer2013 /path/to/fer2013.csv

   # Train model (this will take several hours)
   python model_training.py --epochs 50
   ```

## Step 3: Run the Application

```bash
python app.py
```

The server will start at `http://localhost:5000`

## Step 4: Use the Application

1. Open your browser and go to `http://localhost:5000`

2. **Try Upload**:
   - Click "Upload Image" tab
   - Drag & drop an image or click to browse
   - View the emotion detection results

3. **Try Webcam** (if you have a camera):
   - Click "Use Webcam" tab
   - Click "Start Camera"
   - Click "Capture & Analyze"
   - View real-time emotion detection

4. **Check Statistics**:
   - Click "Statistics" tab
   - See processing statistics

## Troubleshooting

### "Model not found" error
- Make sure you have placed a model file at `models/emotion_model.h5`
- Or train a model using Step 2 Option 2

### Can't install dependencies
```bash
# Try upgrading pip first
pip install --upgrade pip

# Install with verbose output to see errors
pip install -r requirements.txt -v
```

### Webcam not accessible
- Grant camera permissions in your browser
- Close other applications using the camera
- Try Chrome or Firefox browsers

## Test Images

You can test the system with any face image. Here are some sources:
- Use your own photos
- Search for "emotion faces" on free stock photo sites
- Download test images from [Pexels](https://www.pexels.com) or [Unsplash](https://unsplash.com)

## Next Steps

- Read the full [PROJECT_README.md](PROJECT_README.md) for detailed documentation
- Explore the API endpoints for integration
- Train with custom datasets
- Customize the web interface

## Support

If you encounter issues:
1. Check the [PROJECT_README.md](PROJECT_README.md) troubleshooting section
2. Review the console/terminal for error messages
3. Open an issue on GitHub

---

**Happy emotion detecting!**
