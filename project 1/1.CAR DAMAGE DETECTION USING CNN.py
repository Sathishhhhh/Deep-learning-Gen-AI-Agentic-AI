import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import cv2
import io

class CarDamageDetectionCNN:
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path
        if model_path:
            self.load_model()
        else:
            self.build_model()
    
    def build_model(self):
        """Build CNN model for car damage detection"""
        self.model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(2, activation='softmax')  # 2 classes: damaged, not_damaged
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print("Model built successfully!")
    
    def load_model(self):
        """Load pre-trained model"""
        self.model = keras.models.load_model(self.model_path)
        print(f"Model loaded from {self.model_path}")
    
    def preprocess_image(self, image_path_or_bytes):
        """Preprocess image for model prediction"""
        if isinstance(image_path_or_bytes, bytes):
            image = Image.open(io.BytesIO(image_path_or_bytes))
        else:
            image = Image.open(image_path_or_bytes)
        
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        return np.expand_dims(image_array, axis=0)
    
    def predict(self, image_path_or_bytes):
        """Predict car damage"""
        processed_image = self.preprocess_image(image_path_or_bytes)
        prediction = self.model.predict(processed_image)
        
        class_names = ['Not Damaged', 'Damaged']
        class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][class_idx])
        
        return {
            'class': class_names[class_idx],
            'confidence': confidence,
            'predictions': {
                'not_damaged': float(prediction[0][0]),
                'damaged': float(prediction[0][1])
            }
        }
    
    def save_model(self, path):
        """Save trained model"""
        self.model.save(path)
        print(f"Model saved to {path}")

# Example usage
if __name__ == "__main__":
    detector = CarDamageDetectionCNN()
    print(detector.model.summary())

