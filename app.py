from flask import Flask, render_template, request, jsonify
import os
import librosa
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Try to load the model if it exists
model = None
try:
    print("Attempting to load model from:", os.path.abspath('audio_model.h5'))
    model = tf.keras.models.load_model('audio_model.h5')
    print("Model loaded successfully!")
    print("\nModel Summary:")
    model.summary()
    print("\nInput shape:", model.input_shape)
    print("Output shape:", model.output_shape)
except Exception as e:
    print("Warning: Error loading model:", str(e))
    print("Please train the model first using model.py")

# Class labels
class_labels = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling',
                'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

def extract_features(audio_path):
    try:
        print(f"\nProcessing file: {audio_path}")
        
        # Load audio file using librosa (this matches the training process)
        audio, sample_rate = librosa.load(audio_path, duration=3, offset=0.5, sr=22050)
        print(f"Audio loaded, sample rate: {sample_rate}, length: {len(audio)}")
        
        # Extract MFCC features (exactly as in training)
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        print(f"MFCC features extracted, shape: {mfccs_features.shape}")
        
        # Scale the features
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        print(f"Scaled features shape: {mfccs_scaled_features.shape}")  # Should be (40,)
        
        # Reshape for model input: (batch_size, features, channels)
        mfccs_scaled_features = np.expand_dims(mfccs_scaled_features, axis=0)  # Add batch dimension
        mfccs_scaled_features = np.expand_dims(mfccs_scaled_features, axis=2)  # Add channel dimension
        print(f"Final features shape: {mfccs_scaled_features.shape}")  # Should be (1, 40, 1)
        
        return mfccs_scaled_features
    except Exception as e:
        print(f"Error in feature extraction: {str(e)}")
        raise

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 503
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Extract features from the audio file
            features = extract_features(filepath)
            print(f"\nFeatures shape before prediction: {features.shape}")
            
            # Make prediction
            prediction = model.predict(features, verbose=1)
            print(f"\nPrediction shape: {prediction.shape}")
            print(f"Prediction values: {prediction[0]}")
            
            predicted_class = class_labels[np.argmax(prediction[0])]
            confidence = float(np.max(prediction[0]))
            
            # Clean up
            os.remove(filepath)
            
            return jsonify({
                'class': predicted_class,
                'confidence': confidence
            })
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return jsonify({'error': str(e)}), 500
        
@app.route("/hello")
def hello():
    return "Hello, Flask!"

if __name__ == "__main__":
    app.run(debug=True)
