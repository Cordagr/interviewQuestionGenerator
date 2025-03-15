# Standard Library Imports
import os
import time

# Third-Party Library Imports
import numpy as np
import pandas as pd
import kaggle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


# Set up Kaggle API credentials path
os.environ["KAGGLE_CONFIG_DIR"] = os.path.expanduser("~/.kaggle")

# Create dataset directory if it doesn't exist
os.makedirs("./dataset", exist_ok=True)

# Initialize dataframe variable
dataframe = None

# Download dataset
dataset_name = "syedmharis/software-engineering-interview-questions-dataset"
print(f"Downloading dataset: {dataset_name}")

try:
    # Download and extract dataset
    kaggle.api.dataset_download_files(dataset_name, path="./dataset", unzip=True)
    print("Download completed successfully")
    
    time.sleep(2)  # Allow time for extraction
    
    # List all files
    files = os.listdir("./dataset")
    print(f"Files in dataset directory: {files}")
    
    # Find CSV file
    csv_file = next((f for f in files if f.endswith('.csv')), None)
    if csv_file:
        csv_path = f"./dataset/{csv_file}"
        print(f"Attempting to load: {csv_path}")
        
        # Try different encodings to fix the decoding issue
        try:
            dataframe = pd.read_csv(csv_path, encoding="utf-8")
        except UnicodeDecodeError:
            dataframe = pd.read_csv(csv_path, encoding="ISO-8859-1")

        print(f"Loaded dataframe with shape: {dataframe.shape}")
    else:
        raise FileNotFoundError("No CSV file found in dataset directory")
        
except Exception as e:
    print(f"Error downloading or processing dataset: {e}")
    raise ValueError("Failed to load dataframe from dataset")

# Ensure the 'Question' column exists
if 'Question' not in dataframe.columns:
    raise ValueError("The 'Question' column is missing from the dataset.")

# Create feature vectors using TF-IDF
vectorizer = TfidfVectorizer(max_features=60)
X = vectorizer.fit_transform(dataframe['Question'].astype(str)).toarray()

# Ensure the target column exists
target_column = dataframe.columns[-1]  # Assuming last column is the label
Y = dataframe[target_column].astype(str).values

# Encode class values as integers
encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(Y)

# Define model function
def create_baseline():
    model = Sequential([
        Dense(60, input_shape=(60,), activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Wrap model in KerasClassifier
estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)

# Perform Stratified K-Fold Cross-Validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold, scoring="accuracy")

# Print Results
print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
