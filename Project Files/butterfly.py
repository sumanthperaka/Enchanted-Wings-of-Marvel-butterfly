from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import logging
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import kaggle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import pickle
import threading
import time
from werkzeug.utils import secure_filename
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class ButterflyClassifier:
    def __init__(self, model_path='models', dataset_path='dataset'):
        """Initialize the Butterfly Classifier"""
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.model = None
        self.label_encoder = None
        self.class_names = []
        self.history = None
        self.is_training = False
        
        # Create directories if they don't exist
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.dataset_path, exist_ok=True)
        
        # Enhanced butterfly species data
        self.butterfly_data = {
            'ADONIS': {
                'name': 'Adonis Blue',
                'scientific_name': 'Lysandra bellargus',
                'info': 'A small butterfly with brilliant blue wings found in chalk grasslands.',
                'habitat': 'Chalk downs, limestone grasslands',
                'wingspan': '30-36mm',
                'flight_period': 'May-September'
            },
            'AFRICAN GIANT SWALLOWTAIL': {
                'name': 'African Giant Swallowtail',
                'scientific_name': 'Papilio antimachus',
                'info': 'One of the largest butterflies in Africa, with distinctive yellow and black markings.',
                'habitat': 'Tropical rainforests of Central and West Africa',
                'wingspan': '200-230mm',
                'flight_period': 'Year-round in tropical regions'
            },
            'AMERICAN SNOOT': {
                'name': 'American Snoot',
                'scientific_name': 'Libytheana carinenta',
                'info': 'Distinguished by its unusually long labial palps that resemble a snout.',
                'habitat': 'Open woodlands, parks, gardens',
                'wingspan': '38-48mm',
                'flight_period': 'March-October'
            },
            'MONARCH': {
                'name': 'Monarch Butterfly',
                'scientific_name': 'Danaus plexippus',
                'info': 'Famous for their incredible migration journey spanning thousands of miles.',
                'habitat': 'Open fields, meadows, gardens, roadsides',
                'wingspan': '88-102mm',
                'flight_period': 'March-October'
            },
            'BLUE MORPHO': {
                'name': 'Blue Morpho',
                'scientific_name': 'Morpho menelaus',
                'info': 'Large tropical butterfly with brilliant metallic blue wings that shimmer in sunlight.',
                'habitat': 'Tropical rainforests, forest edges',
                'wingspan': '120-200mm',
                'flight_period': 'Year-round in tropical regions'
            },
            'CABBAGE WHITE': {
                'name': 'Cabbage White',
                'scientific_name': 'Pieris rapae',
                'info': 'Small white butterfly with black spots. Common garden visitor.',
                'habitat': 'Gardens, fields, roadsides, parks',
                'wingspan': '45-55mm',
                'flight_period': 'March-October'
            },
            'PAINTED LADY': {
                'name': 'Painted Lady',
                'scientific_name': 'Vanessa cardui',
                'info': 'Orange and black butterfly with white spots. Known for long-distance migrations.',
                'habitat': 'Open areas, gardens, fields, deserts',
                'wingspan': '50-65mm',
                'flight_period': 'March-October'
            },
            'RED ADMIRAL': {
                'name': 'Red Admiral',
                'scientific_name': 'Vanessa atalanta',
                'info': 'Dark butterfly with distinctive red bands and white spots.',
                'habitat': 'Gardens, parks, woodlands, coastal areas',
                'wingspan': '56-64mm',
                'flight_period': 'April-October'
            },
            'SWALLOWTAIL': {
                'name': 'Old World Swallowtail',
                'scientific_name': 'Papilio machaon',
                'info': 'Large yellow butterfly with black markings and distinctive tail streamers.',
                'habitat': 'Fenland, chalk downs, mountains',
                'wingspan': '65-86mm',
                'flight_period': 'April-September'
            },
            'TIGER SWALLOWTAIL': {
                'name': 'Eastern Tiger Swallowtail',
                'scientific_name': 'Papilio glaucus',
                'info': 'Large yellow butterfly with black tiger stripes and blue spots on hindwings.',
                'habitat': 'Deciduous forests, parks, gardens',
                'wingspan': '79-140mm',
                'flight_period': 'March-November'
            }
        }
    
    def setup_kaggle_api(self):
        """Setup Kaggle API credentials"""
        try:
            kaggle_dir = os.path.expanduser('~/.kaggle')
            if not os.path.exists(kaggle_dir):
                os.makedirs(kaggle_dir)
            
            kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
            if not os.path.exists(kaggle_json):
                logging.warning("Kaggle API credentials not found. Please set up kaggle.json")
                return False
            
            os.chmod(kaggle_json, 0o600)
            return True
        except Exception as e:
            logging.error(f"Error setting up Kaggle API: {e}")
            return False
    
    def download_dataset(self, dataset_name='gpiosenka/butterfly-images40-species'):
        """Download butterfly dataset from Kaggle"""
        try:
            if not self.setup_kaggle_api():
                return False
            
            logging.info(f"Downloading dataset: {dataset_name}")
            kaggle.api.dataset_download_files(dataset_name, path=self.dataset_path, unzip=True)
            
            logging.info("Dataset downloaded successfully!")
            return True
            
        except Exception as e:
            logging.error(f"Error downloading dataset: {e}")
            return False
    
    def load_and_preprocess_data(self, img_size=(224, 224)):
        """Load and preprocess butterfly images"""
        try:
            images = []
            labels = []
            
            # Walk through directory structure
            for root, dirs, files in os.walk(self.dataset_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        # Get label from parent directory name
                        label = os.path.basename(root)
                        
                        # Skip if it's the root dataset directory
                        if label == os.path.basename(self.dataset_path):
                            continue
                        
                        # Load and preprocess image
                        img_path = os.path.join(root, file)
                        try:
                            img = cv2.imread(img_path)
                            if img is not None:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                img = cv2.resize(img, img_size)
                                img = img.astype('float32') / 255.0
                                
                                images.append(img)
                                labels.append(label)
                        except Exception as e:
                            logging.warning(f"Error processing image {img_path}: {e}")
                            continue
            
            if len(images) == 0:
                logging.error("No images found in dataset directory")
                return None, None
            
            images = np.array(images)
            labels = np.array(labels)
            
            logging.info(f"Loaded {len(images)} images with {len(np.unique(labels))} classes")
            
            return images, labels
            
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return None, None
    
    def create_cnn_model(self, num_classes, input_shape=(224, 224, 3)):
        """Create a CNN model for butterfly classification"""
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Fourth Convolutional Block
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Fifth Convolutional Block
            Conv2D(512, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Classifier
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model_async(self, epochs=30, batch_size=16):
        """Train the model asynchronously"""
        def train():
            self.is_training = True
            try:
                # Check if dataset exists, download if necessary
                if not os.path.exists(self.dataset_path) or not os.listdir(self.dataset_path):
                    logging.info("Dataset not found. Downloading...")
                    if not self.download_dataset():
                        logging.error("Failed to download dataset")
                        self.is_training = False
                        return False
                
                # Load and preprocess data
                images, labels = self.load_and_preprocess_data()
                if images is None:
                    self.is_training = False
                    return False
                
                # Encode labels
                self.label_encoder = LabelEncoder()
                encoded_labels = self.label_encoder.fit_transform(labels)
                self.class_names = self.label_encoder.classes_
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    images, encoded_labels, test_size=0.2, 
                    random_state=42, stratify=encoded_labels
                )
                
                # Create model
                num_classes = len(self.class_names)
                self.model = self.create_cnn_model(num_classes)
                
                # Data augmentation
                datagen = ImageDataGenerator(
                    rotation_range=20,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    horizontal_flip=True,
                    zoom_range=0.2,
                    brightness_range=[0.8, 1.2],
                    fill_mode='nearest'
                )
                
                # Callbacks
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                )
                
                reduce_lr = ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=5,
                    min_lr=0.0001,
                    verbose=1
                )
                
                checkpoint = ModelCheckpoint(
                    os.path.join(self.model_path, 'best_model.h5'),
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
                
                # Train model
                logging.info("Starting model training...")
                self.history = self.model.fit(
                    datagen.flow(X_train, y_train, batch_size=batch_size),
                    epochs=epochs,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping, reduce_lr, checkpoint],
                    verbose=1
                )
                
                # Save model and label encoder
                self.save_model()
                
                logging.info("Model trained and saved successfully!")
                self.is_training = False
                return True
                
            except Exception as e:
                logging.error(f"Error training model: {e}")
                self.is_training = False
                return False
        
        # Start training in a separate thread
        thread = threading.Thread(target=train)
        thread.daemon = True
        thread.start()
        
        return True
    
    def save_model(self):
        """Save the trained model and label encoder"""
        try:
            # Save model
            model_file = os.path.join(self.model_path, 'butterfly_classifier.h5')
            self.model.save(model_file)
            
            # Save label encoder
            encoder_file = os.path.join(self.model_path, 'label_encoder.pkl')
            with open(encoder_file, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            
            # Save class names
            classes_file = os.path.join(self.model_path, 'class_names.pkl')
            with open(classes_file, 'wb') as f:
                pickle.dump(self.class_names, f)
            
            logging.info("Model and encoders saved successfully!")
            
        except Exception as e:
            logging.error(f"Error saving model: {e}")
    
    def load_model(self):
        """Load the trained model and label encoder"""
        try:
            model_file = os.path.join(self.model_path, 'butterfly_classifier.h5')
            encoder_file = os.path.join(self.model_path, 'label_encoder.pkl')
            classes_file = os.path.join(self.model_path, 'class_names.pkl')
            
            if os.path.exists(model_file) and os.path.exists(encoder_file):
                self.model = load_model(model_file)
                
                with open(encoder_file, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                
                if os.path.exists(classes_file):
                    with open(classes_file, 'rb') as f:
                        self.class_names = pickle.load(f)
                else:
                    self.class_names = self.label_encoder.classes_
                
                logging.info("Model loaded successfully!")
                return True
            else:
                logging.warning("No trained model found.")
                return False
                
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return False
    
    def preprocess_image(self, image_data, img_size=(224, 224)):
        """Preprocess a single image for prediction"""
        try:
            # Convert bytes to numpy array
            img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize and normalize
            img = cv2.resize(img, img_size)
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            
            return img
            
        except Exception as e:
            logging.error(f"Error preprocessing image: {e}")
            return None
    
    def predict(self, image_data, top_k=5):
        """Predict butterfly species from image data"""
        try:
            if self.model is None:
                if not self.load_model():
                    logging.error("No trained model available")
                    return None
            
            # Preprocess image
            img_array = self.preprocess_image(image_data)
            if img_array is None:
                return None
            
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)
            
            # Get top predictions
            top_indices = np.argsort(predictions[0])[::-1][:top_k]
            top_predictions = []
            
            for idx in top_indices:
                species_code = self.label_encoder.inverse_transform([idx])[0]
                confidence = predictions[0][idx]
                
                # Get species information
                species_info = self.butterfly_data.get(species_code.upper(), {
                    'name': species_code,
                    'scientific_name': 'Unknown',
                    'info': 'No information available',
                    'habitat': 'Unknown',
                    'wingspan': 'Unknown',
                    'flight_period': 'Unknown'
                })
                
                top_predictions.append({
                    'species_code': species_code,
                    'species_name': species_info['name'],
                    'scientific_name': species_info['scientific_name'],
                    'confidence': float(confidence),
                    'info': species_info['info'],
                    'habitat': species_info['habitat'],
                    'wingspan': species_info['wingspan'],
                    'flight_period': species_info['flight_period']
                })
            
            return top_predictions
            
        except Exception as e:
            logging.error(f"Error making prediction: {e}")
            return None

# Initialize classifier
classifier = ButterflyClassifier()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS