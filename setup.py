"""
Setup script for Facial Emotion Recognition System
Helps users quickly set up the project
"""

import os
import sys
import subprocess


def print_step(step_num, message):
    """Print a formatted step message"""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {message}")
    print(f"{'='*60}\n")


def check_python_version():
    """Check if Python version is compatible"""
    print_step(1, "Checking Python Version")

    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("ERROR: Python 3.8 or higher is required!")
        return False

    print("Python version is compatible!")
    return True


def create_directories():
    """Create necessary directories"""
    print_step(2, "Creating Project Directories")

    directories = [
        'datasets/training/angry',
        'datasets/training/disgust',
        'datasets/training/fear',
        'datasets/training/happy',
        'datasets/training/sad',
        'datasets/training/surprise',
        'datasets/training/neutral',
        'datasets/validation/angry',
        'datasets/validation/disgust',
        'datasets/validation/fear',
        'datasets/validation/happy',
        'datasets/validation/sad',
        'datasets/validation/surprise',
        'datasets/validation/neutral',
        'datasets/uploads',
        'models',
        'static/css',
        'static/js',
        'templates'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created: {directory}")

    print("\nAll directories created successfully!")
    return True


def check_virtualenv():
    """Check if running in virtual environment"""
    print_step(3, "Checking Virtual Environment")

    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )

    if in_venv:
        print("Running in virtual environment")
        return True
    else:
        print("WARNING: Not running in a virtual environment!")
        print("It's recommended to use a virtual environment.")
        response = input("Continue anyway? (y/n): ")
        return response.lower() == 'y'


def install_dependencies():
    """Install required packages"""
    print_step(4, "Installing Dependencies")

    print("This may take several minutes...")
    print("Installing packages from requirements.txt...\n")

    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ])
        print("\nDependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Failed to install dependencies: {e}")
        return False


def check_model():
    """Check if model exists"""
    print_step(5, "Checking for Trained Model")

    model_path = 'models/emotion_model.h5'

    if os.path.exists(model_path):
        print(f"Model found at: {model_path}")
        return True
    else:
        print(f"No model found at: {model_path}")
        print("\nYou have two options:")
        print("1. Download a pre-trained model and place it in models/emotion_model.h5")
        print("2. Train your own model using model_training.py")
        print("\nSee QUICKSTART.md for detailed instructions.")
        return False


def print_next_steps(model_exists):
    """Print next steps for the user"""
    print_step(6, "Setup Complete!")

    print("Next steps:\n")

    if not model_exists:
        print("1. Get a trained model:")
        print("   - Download a pre-trained model OR")
        print("   - Train your own: python model_training.py")
        print()

    print(f"{'1' if model_exists else '2'}. Run the application:")
    print("   python app.py")
    print()

    print(f"{'2' if model_exists else '3'}. Open your browser:")
    print("   http://localhost:5000")
    print()

    print("For detailed instructions, see:")
    print("- QUICKSTART.md (quick start guide)")
    print("- PROJECT_README.md (full documentation)")
    print()

    print("Happy emotion detecting!")


def main():
    """Main setup function"""
    print("\n" + "="*60)
    print("FACIAL EMOTION RECOGNITION SYSTEM - SETUP")
    print("="*60)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Create directories
    if not create_directories():
        print("ERROR: Failed to create directories")
        sys.exit(1)

    # Check virtual environment
    if not check_virtualenv():
        print("Setup cancelled by user")
        sys.exit(0)

    # Install dependencies
    if not install_dependencies():
        print("ERROR: Failed to install dependencies")
        sys.exit(1)

    # Check for model
    model_exists = check_model()

    # Print next steps
    print_next_steps(model_exists)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        sys.exit(1)
