# %%
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import gc
import os
import datetime
import time
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import json
import string
from collections import Counter
from liblinear.liblinearutil import *
import scipy.sparse as sp
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer

# %% [markdown]
# # Preprocessing

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)

# Define paths
train_root = 'data/yelp_reviews_train.json'
test_root = 'data/yelp_reviews_test.json'
val_root = 'data/yelp_reviews_dev.json'
stopword_root = 'data/stopword.list'

# Token pattern
TOKEN_PATTERN = re.compile(r'\b[a-z]+\b')

with open(stopword_root, 'r') as f:
    stopwords = {line.strip() for line in f}
print(f"Number of stopwords: {len(stopwords)}")

translator = str.maketrans('', '', string.punctuation)

# %%
class YelpReviewsDataset(Dataset):
    def __init__(self, features, labels=None):
        """
        Custom Dataset for Yelp reviews using precomputed TF-IDF features.

        Args:
            features (np.ndarray): Precomputed TF-IDF feature matrix.
            labels (np.ndarray, optional): Corresponding labels (class indices).
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.long)  # Use long for class indices
        else:
            self.labels = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        else:
            return self.features[idx]  # Return only features if no labels are provided

# %%
def clean_and_tokenize(text, stopwords, translator, token_pattern):
    text = text.lower().translate(translator)
    tokens = token_pattern.findall(text)
    return [token for token in tokens if token not in stopwords]

# Function to load training data and extract texts and stars
def load_training_data(file_path):
    texts, stars = [], []
    with open(file_path, 'r') as f:
        for line in tqdm(f, desc=f"Loading data from {file_path}"):
            review = json.loads(line)
            texts.append(review['text'])
            stars.append(review['stars'])
    return texts, np.array(stars)

# Function to load text data (for test and validation sets)
def load_text_data(file_path):
    texts = []
    with open(file_path, 'r') as f:
        for line in tqdm(f, desc=f"Loading data from {file_path}"):
            review = json.loads(line)
            texts.append(review['text'])
    return texts

# Function to tokenize text data
def tokenize_texts(texts, stopwords, translator, token_pattern):
    return [clean_and_tokenize(text, stopwords, translator, token_pattern) for text in tqdm(texts, desc="Tokenizing texts")]


# Define function to create TF-IDF feature matrix
def create_tfidf_matrix(texts, max_features=2000):
    vectorizer = TfidfVectorizer(
        stop_words=stopwords,  # Use the loaded stopwords
        max_features=max_features,  # Maximum number of features to keep
        lowercase=True,  # Convert texts to lowercase
        token_pattern=r'\b[a-z]+\b'  # Token pattern to use for extracting words
    )
    # Fit the vectorizer on the texts and return the resulting TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(texts).toarray()
    return tfidf_matrix, vectorizer

# Load and preprocess the texts
train_texts, stars = load_training_data(train_root)
val_texts = load_text_data(val_root)
test_texts = load_text_data(test_root)

print("Making the TF-IDF matrix...")

# Create the TF-IDF feature matrix for train, val, and test sets
features_tfidf_train, tfidf_vectorizer = create_tfidf_matrix(train_texts, max_features=2000)
features_tfidf_val = tfidf_vectorizer.transform(val_texts).toarray()
features_tfidf_test = tfidf_vectorizer.transform(test_texts).toarray()

# Print the shape of the matrices
print(f"TF-IDF Train Shape: {features_tfidf_train.shape}")
print(f"TF-IDF Val Shape: {features_tfidf_val.shape}")
print(f"TF-IDF Test Shape: {features_tfidf_test.shape}")

# %%
# Convert stars to labels
# Adjust star ratings from 1-5 to 0-4
labels_train = stars - 1  # Assuming stars are [1, 2, 3, 4, 5]
#labels_val = one_hot_encode_labels(stars[len(features_tfidf_train):])  # Validation labels

# Create datasets for training, validation, and test
train_dataset = YelpReviewsDataset(features=features_tfidf_train, labels=labels_train)  # Training set with labels
val_dataset = YelpReviewsDataset(features=features_tfidf_val)  # Validation set without labels
test_dataset = YelpReviewsDataset(features=features_tfidf_test)  # Test set without labels

# Create data loaders
batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Train features shape: {features_tfidf_train.shape}")
print(f"Train labels shape: {labels_train.shape}")

# %%
print(f"Train features shape: {features_tfidf_train.shape}")
print(f"Train labels shape: {labels_train.shape}")
print(f"Validation features shape: {features_tfidf_val.shape}")

# %%
# Define your model with a hidden layer
class RMLR(nn.Module):
    def __init__(self, num_features, num_classes):
        super(RMLR, self).__init__()
        self.hidden = nn.Linear(num_features, 256)
        self.relu = nn.ReLU()
        self.output = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

model = RMLR(num_features=features_tfidf_train.shape[1], num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

# Update batch size and epochs
batch_size = 128
epochs = 20

# Training loop with RMSE calculation
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total_samples = 0
    squared_error = 0.0  # For RMSE calculation
    
    for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * features.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        
        # Compute soft predictions (expected ratings)
        probabilities = nn.functional.softmax(outputs, dim=1)
        class_indices = torch.arange(1, 6).float().to(device)  # Ratings from 1 to 5
        expected_ratings = torch.matmul(probabilities, class_indices)
        true_ratings = labels.float() + 1  # Adjust labels back to 1-5
        
        # Accumulate squared errors
        squared_error += torch.sum((expected_ratings - true_ratings) ** 2).item()
    
    epoch_loss = running_loss / total_samples
    epoch_accuracy = 100 * correct / total_samples
    rmse = np.sqrt(squared_error / total_samples)
    
    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, RMSE: {rmse:.4f}")

# %%
# Generate predictions for validation (dev) and test sets
def generate_predictions(model, data_loader, output_file):
    model.eval()
    hard_predictions = []
    soft_predictions = []

    with torch.no_grad():
        for features in tqdm(data_loader, desc="Generating predictions"):
            features = features.to(device)
            outputs = model(features)
            probabilities = nn.functional.softmax(outputs, dim=1)

            # Hard predictions
            _, predicted = torch.max(outputs.data, 1)
            hard_predictions.extend((predicted + 1).cpu().numpy())  # Adjusting back to 1-5

            # Soft predictions
            class_indices = torch.arange(1, 6).float().to(device)
            soft_pred = torch.matmul(probabilities, class_indices)
            soft_predictions.extend(soft_pred.cpu().numpy())

    # Write predictions to file
    with open(output_file, 'w') as f:
        for hard_pred, soft_pred in zip(hard_predictions, soft_predictions):
            f.write(f"{hard_pred} {soft_pred:.4f}\n")

# Generate predictions for validation (dev) set
generate_predictions(model, val_loader, "dev-predictions.txt")

# Generate predictions for test set
generate_predictions(model, test_loader, "test-predictions.txt")


