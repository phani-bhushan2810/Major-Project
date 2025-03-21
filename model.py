import numpy as np
import pandas as pd
import os
import librosa
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Audio feature extraction function
def features_extractor(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        return mfccs_scaled_features
    except Exception as e:
        print(f"Error extracting features from {file_name}: {str(e)}")
        return None

# Load metadata
audio_dataset_path = 'static/audio/'
metadata = pd.read_csv('static/metadata/UrbanSound8K.csv')

print("Loading dataset...")
print(f"Total files in dataset: {len(metadata)}")

# Extract features from all audio files
extracted_features = []
for index_num, row in tqdm(metadata.iterrows(), total=len(metadata)):
    file_name = os.path.join(os.path.abspath(audio_dataset_path), 'fold'+str(row["fold"])+'/', str(row["slice_file_name"]))
    final_class_labels = row["class"]
    data = features_extractor(file_name)
    if data is not None:
        extracted_features.append([data, final_class_labels])
    if index_num % 100 == 0:
        print(f"Processed {index_num} files...")

print(f"Successfully extracted features from {len(extracted_features)} files")

# Convert to DataFrame
extracted_features_df = pd.DataFrame(extracted_features, columns=['feature', 'class'])
print("Feature extraction completed")

# Split the dataset
X = np.array(extracted_features_df['feature'].tolist())
y = np.array(extracted_features_df['class'].tolist())

# Encode the labels
le = LabelEncoder()
y = le.fit_transform(y)

print("Preparing training data...")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape X for CNN input
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print("Building and training model...")

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(40, 1)),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Training model...")

# Train the model
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=50,
                    batch_size=32,
                    verbose=1)

# Save the model
model.save('audio_model.h5')

print("\nModel training completed and saved as 'audio_model.h5'")

# Print final accuracy
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal test accuracy: {test_accuracy*100:.2f}%")
