"""
Model Training Module
Trains the emotion detection model using the dataset
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import cv2


class EmotionModelTrainer:
    """
    Class for training emotion detection models
    """

    def __init__(self, img_size=(48, 48), num_classes=7):
        """
        Initialize the model trainer

        Args:
            img_size (tuple): Size of input images (height, width)
            num_classes (int): Number of emotion classes
        """
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None

    def build_model(self):
        """
        Build a CNN model for emotion detection
        """
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                         input_shape=(self.img_size[0], self.img_size[1], 1)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            # Fully Connected Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            # Output Layer
            layers.Dense(self.num_classes, activation='softmax')
        ])

        self.model = model
        return model

    def compile_model(self, learning_rate=0.001):
        """
        Compile the model with optimizer and loss function

        Args:
            learning_rate (float): Learning rate for optimizer
        """
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def create_data_generators(self, train_dir, val_dir, batch_size=32):
        """
        Create data generators for training and validation

        Args:
            train_dir (str): Directory containing training data
            val_dir (str): Directory containing validation data
            batch_size (int): Batch size for training

        Returns:
            Tuple of (train_generator, val_generator)
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )

        # Only rescaling for validation
        val_datagen = ImageDataGenerator(rescale=1./255)

        # Create generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            color_mode='grayscale',
            class_mode='categorical'
        )

        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            color_mode='grayscale',
            class_mode='categorical'
        )

        return train_generator, val_generator

    def train(self, train_generator, val_generator, epochs=50, model_save_path='models/emotion_model.h5'):
        """
        Train the model

        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs (int): Number of training epochs
            model_save_path (str): Path to save the trained model

        Returns:
            Training history
        """
        # Create callbacks
        callbacks = [
            # Save best model
            ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),

            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),

            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]

        # Train the model
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def plot_training_history(self, history, save_path='training_history.png'):
        """
        Plot training history

        Args:
            history: Training history object
            save_path (str): Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plot accuracy
        axes[0].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)

        # Plot loss
        axes[1].plot(history.history['loss'], label='Train Loss')
        axes[1].plot(history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")


def prepare_fer2013_dataset(csv_path, output_dir):
    """
    Prepare FER2013 dataset from CSV file
    FER2013 is a popular emotion detection dataset
    Download from: https://www.kaggle.com/datasets/msambare/fer2013

    Args:
        csv_path (str): Path to fer2013.csv
        output_dir (str): Output directory for organized dataset
    """
    import pandas as pd

    # Read CSV
    df = pd.read_csv(csv_path)

    # Emotion mapping
    emotion_labels = {
        0: 'angry',
        1: 'disgust',
        2: 'fear',
        3: 'happy',
        4: 'sad',
        5: 'surprise',
        6: 'neutral'
    }

    # Create directories
    for split in ['Training', 'PublicTest', 'PrivateTest']:
        for emotion in emotion_labels.values():
            os.makedirs(os.path.join(output_dir, split, emotion), exist_ok=True)

    # Process images
    for idx, row in df.iterrows():
        emotion = emotion_labels[row['emotion']]
        usage = row['Usage']
        pixels = np.array([int(p) for p in row['pixels'].split()]).reshape(48, 48)

        # Save image
        img_path = os.path.join(output_dir, usage, emotion, f'{idx}.jpg')
        cv2.imwrite(img_path, pixels)

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} images...")

    print(f"Dataset prepared successfully in {output_dir}")


def train_new_model(train_dir='datasets/training', val_dir='datasets/validation',
                   epochs=50, batch_size=32, model_save_path='models/emotion_model.h5'):
    """
    Train a new emotion detection model

    Args:
        train_dir (str): Directory containing training data
        val_dir (str): Directory containing validation data
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        model_save_path (str): Path to save the trained model
    """
    print("Initializing model trainer...")
    trainer = EmotionModelTrainer()

    print("Building model...")
    trainer.build_model()
    trainer.model.summary()

    print("Compiling model...")
    trainer.compile_model()

    print("Creating data generators...")
    train_gen, val_gen = trainer.create_data_generators(train_dir, val_dir, batch_size)

    print(f"Training classes: {train_gen.class_indices}")
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")

    print("\nStarting training...")
    history = trainer.train(train_gen, val_gen, epochs, model_save_path)

    print("\nPlotting training history...")
    trainer.plot_training_history(history)

    print(f"\nTraining complete! Model saved to {model_save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train emotion detection model')
    parser.add_argument('--train-dir', default='datasets/training',
                       help='Training data directory')
    parser.add_argument('--val-dir', default='datasets/validation',
                       help='Validation data directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--model-path', default='models/emotion_model.h5',
                       help='Path to save the trained model')
    parser.add_argument('--prepare-fer2013', type=str,
                       help='Path to fer2013.csv to prepare dataset')

    args = parser.parse_args()

    # Prepare dataset if requested
    if args.prepare_fer2013:
        print("Preparing FER2013 dataset...")
        prepare_fer2013_dataset(args.prepare_fer2013, 'datasets')

    # Train model
    train_new_model(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_save_path=args.model_path
    )
