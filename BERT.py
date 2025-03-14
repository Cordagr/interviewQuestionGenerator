# Standard Library Imports
import re
import random

# Third-Party Library Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PyTorch
import torch

# TensorFlow & Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Sci-Keras (for TensorFlow & Scikit-Learn compatibility)
from scikeras.wrappers import KerasClassifier

# Scikit-Learn Imports
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# Transformers for NLP tasks
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
)


import kaggle

# Set up Kaggle API
import os

os.environ["KAGGLE_CONFIG_DIR"] = os.path.expanduser("~/.kaggle")

# Download dataset (Make sure you have kaggle.json in ~/.kaggle)
dataset_name = "syedmharis/software-engineering-interview-questions-dataset"


# Load dataset
dataframe = pd.read_csv("./dataset/dataset.csv")  # Update filename accordingly


dataset = dataframe.values

# Split into features (X) and labels (Y)
X = dataset[:, 0:60].astype(float)
Y = dataset[:, 60]

# Encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)


def create_baseline():
    model = Sequential()
    model.add(Dense(60, input_shape=(60,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


estimator = KerasClassifier(model=create_baseline, epochs=100, batch_size=5, verbose=0)

# Perform Stratified K-Fold Cross-Validation
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)

# Print Results
print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
