
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
from rmlr123 import rmlr123
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

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

# Function to count document frequencies (DF) or collection term frequencies (CTF)
def count_frequencies(tokenized_texts, use_unique_tokens=False):
    token_counter = Counter()
    for tokens in tqdm(tokenized_texts, desc="Counting tokens"):
        if use_unique_tokens:
            tokens = set(tokens)  # Unique tokens for DF
        token_counter.update(tokens)
    return token_counter

def create_feature_matrix(tokenized_texts, token_to_idx, num_features, binary=False):
    """
    Creates a dense feature matrix for the input data.

    Args:
        tokenized_texts (list of list of str): Tokenized texts.
        token_to_idx (dict): Token to index mapping.
        num_features (int): Number of features (e.g., vocabulary size).
        binary (bool): If True, use binary presence; else use term frequencies.

    Returns:
        np.ndarray: Dense feature matrix of shape (num_documents, num_features).
    """
    num_documents = len(tokenized_texts)
    feature_matrix = np.zeros((num_documents, num_features), dtype=np.float32)
    
    for doc_idx, tokens in enumerate(tqdm(tokenized_texts, desc="Creating feature matrix")):
        token_counts = Counter(tokens)
        for token, count in token_counts.items():
            if token in token_to_idx:
                token_idx = token_to_idx[token]
                if binary:
                    feature_matrix[doc_idx, token_idx] = 1.0
                else:
                    feature_matrix[doc_idx, token_idx] += count
    return feature_matrix

# Function to display top tokens and star rating distribution
def display_statistics(tokenized_texts, stars):
    # Counting token frequencies
    print("\nCounting token frequencies...")
    token_counter = Counter()
    for tokens in tqdm(tokenized_texts, desc="Counting token frequencies"):
        token_counter.update(tokens)

    # Top 9 most frequent tokens
    top_9 = token_counter.most_common(9)
    print("\nTop 9 Most Frequent Tokens:")
    print("{:<5} {:<15} {:<10}".format('Rank', 'Token', 'Count'))
    for rank, (token, count) in enumerate(top_9, start=1):
        print("{:<5} {:<15} {:<10}".format(rank, token, count))

    # Star rating distribution
    star_distribution = Counter(stars)
    total_reviews = len(stars)
    
    print("\nStar Rating Distribution:")
    print("{:<5} {:<10} {:<10}".format('Star', 'Count', 'Percentage (%)'))
    for star, count in sorted(star_distribution.items()):
        percentage = (count / total_reviews) * 100
        print("{:<5} {:<10} {:<10.2f}".format(star, count, percentage))

# %%
# Main processing function
def main_processing(train_root, test_root, val_root, stopwords, translator, token_pattern):
    # Load training, test, and dev data
    train_texts, stars = load_training_data(train_root)
    test_texts = load_text_data(test_root)
    dev_texts = load_text_data(val_root)

    # Tokenize texts
    tokenized_train = tokenize_texts(train_texts, stopwords, translator, token_pattern)
    tokenized_test = tokenize_texts(test_texts, stopwords, translator, token_pattern)
    tokenized_dev = tokenize_texts(dev_texts, stopwords, translator, token_pattern)

    # Count document frequencies (DF)
    print("Counting document frequencies (DF)...")
    df_counter = count_frequencies(tokenized_train, use_unique_tokens=True)
    top_2000_df = [token for token, _ in df_counter.most_common(2000)]
    token_to_idx_df = {token: idx for idx, token in enumerate(top_2000_df)}

    # Create sparse DF feature matrices for train, dev, and test sets
    print("Creating sparse DF feature matrices...")
    num_features = len(top_2000_df)
    features_df_train = create_feature_matrix(tokenized_train, token_to_idx_df, num_features)
    features_df_dev = create_feature_matrix(tokenized_dev, token_to_idx_df, num_features)
    features_df_test = create_feature_matrix(tokenized_test, token_to_idx_df, num_features)

    # Count collection term frequencies (CTF)
    print("Counting collection term frequencies (CTF)...")
    ctf_counter = count_frequencies(tokenized_train)
    top_2000_ctf = [token for token, _ in ctf_counter.most_common(2000)]
    token_to_idx_ctf = {token: idx for idx, token in enumerate(top_2000_ctf)}

    # Create sparse CTF feature matrices for train, dev, and test sets
    print("Creating sparse CTF feature matrices...")
    features_ctf_train = create_feature_matrix(tokenized_train, token_to_idx_ctf, num_features)
    features_ctf_dev = create_feature_matrix(tokenized_dev, token_to_idx_ctf, num_features)
    features_ctf_test = create_feature_matrix(tokenized_test, token_to_idx_ctf, num_features)

    # Print feature matrix shapes for verification
    print(f"DF feature matrix shapes: train={features_df_train.shape}, dev={features_df_dev.shape}, test={features_df_test.shape}")
    print(f"CTF feature matrix shapes: train={features_ctf_train.shape}, dev={features_ctf_dev.shape}, test={features_ctf_test.shape}")

    return features_df_train, features_df_dev, features_df_test, features_ctf_train, features_ctf_dev, features_ctf_test, stars

# %%
# Call main processing function with appropriate arguments
features_df_train, features_df_dev, features_df_test, features_ctf_train, features_ctf_dev, features_ctf_test, stars = main_processing(
    train_root, test_root, val_root, stopwords, translator, TOKEN_PATTERN
)

# %%
# Training the model with LibLinear
liblinear_params = {
    'solver_type': L2R_LR,  # L2-regularized logistic regression
    'C': 1.0,               # Regularization parameter
    'eps': 0.01,            # Stopping criteria
    'verbose': False        # Disable verbose output
}

# Wrapping LibLinear training and prediction in tqdm

# Train LibLinear model using DF and CTF features
print("Training LibLinear models...")

with tqdm(total=2, desc="Training models") as pbar:
    # Train the model for DF features
    model_df = train(stars, features_df_train, f"-s {liblinear_params['solver_type']} -c {liblinear_params['C']} -e {liblinear_params['eps']} -q")
    pbar.update(1)  # Progress after training DF model
    print("LibLinear model training completed for DF features.")

    # Train the model for CTF features
    model_ctf = train(stars, features_ctf_train, f"-s {liblinear_params['solver_type']} -c {liblinear_params['C']} -e {liblinear_params['eps']} -q")
    pbar.update(1)  # Progress after training CTF model
    print("LibLinear model training completed for CTF features.")

# Making predictions with DF and CTF models
print("Making predictions using LibLinear models...")

with tqdm(total=2, desc="Making predictions") as pbar:
    # Predictions using DF features
    y_hard_train_df, accuracy_train_df, y_soft_train_df = predict(stars, features_df_train, model_df, '-b 1')
    pbar.update(1)  # Progress after DF predictions
    print(f"Training Accuracy (DF Features): {accuracy_train_df[0]}%")

    # Predictions using CTF features
    y_hard_train_ctf, accuracy_train_ctf, y_soft_train_ctf = predict(stars, features_ctf_train, model_ctf, '-b 1')
    pbar.update(1)  # Progress after CTF predictions
    print(f"Training Accuracy (CTF Features): {accuracy_train_ctf[0]}%")


# %%
# Make predictions for the development set using the trained LibLinear models
y_hard_dev_df, _, y_soft_dev_df = predict([], features_df_dev, model_df, '-b 1')
y_hard_dev_ctf, _, y_soft_dev_ctf = predict([], features_ctf_dev, model_ctf, '-b 1')

# %%
def save_predictions(file_name, y_hard, y_soft):
    """
    Saves predictions to a text file in the required format.

    Each line in the file will contain:
    <hard_prediction> <soft_prediction>
    
    Parameters:
    - file_name (str): The name of the output file.
    - y_hard (list of float): List of hard predictions.
    - y_soft (list of list of float): List of soft predictions (probabilities).
    """
    with open(file_name, 'w') as f:
        for hard, soft_probs in zip(y_hard, y_soft):
            hard_int = int(hard)  # Convert hard prediction to integer
            
            # Safely retrieve the soft prediction
            soft = soft_probs[hard_int] if 0 <= hard_int < len(soft_probs) else 0.0
            
            # Write hard and soft predictions in the required format
            f.write(f"{hard_int} {soft:.4f}\n")

# Validate that the predictions have the expected number of lines
def validate_predictions(y_hard, y_soft, required_lines):
    """
    Validates the length of hard and soft predictions.
    
    Parameters:
    - y_hard (list): Hard predictions.
    - y_soft (list): Soft predictions.
    - required_lines (int): Expected number of lines.
    
    Raises an assertion error if the lengths don't match the required number of lines.
    """
    assert len(y_hard) == required_lines, f"Hard predictions should have {required_lines} lines."
    assert len(y_soft) == required_lines, f"Soft predictions should have {required_lines} lines."

# %%
required_dev_lines = 157010

# Validate DF predictions
validate_predictions(y_hard_dev_df, y_soft_dev_df, required_dev_lines)

# Validate CTF predictions
validate_predictions(y_hard_dev_ctf, y_soft_dev_ctf, required_dev_lines)

# Save predictions for Development Set (DF and CTF)
#save_predictions("dev-predictions-df-svm.txt", y_hard_dev_df, y_soft_dev_df)
#save_predictions("dev-predictions-ctf-svm.txt", y_hard_dev_ctf, y_soft_dev_ctf)

print("Prediction files generated successfully.")

# %%
#import importlib
#import rmlr123
#importlib.reload(rmlr123)

#Stop times needing to reload module

# %%
def one_hot_encode_labels(stars, num_classes=5):
    one_hot_labels = np.zeros((len(stars), num_classes))
    for idx, star in enumerate(stars):
        one_hot_labels[idx, star - 1] = 1  # stars are 1-indexed, hence subtracting 1
    return one_hot_labels

# One-hot encode star ratings
y_one_hot = one_hot_encode_labels(stars)

# Train the RMLR model using DF features
num_classes = 5  # For 5-star ratings
learning_rate = 0.01
lambda_reg = 0.1
class_weights = np.ones(num_classes)  # Adjust if you have class imbalance
num_features = features_df_train.shape[1]

model_df = rmlr123(num_features=num_features, num_classes=num_classes,
                   learning_rate=learning_rate, lambda_reg=lambda_reg,
                   class_weights=class_weights)

X_train_df, X_val_df, Y_train_df, Y_val_df = train_test_split(features_df_train, y_one_hot, test_size=0.2, random_state=42)

# Train the model
model_df.train(X_train_df, Y_train_df, X_val=X_val_df, Y_val=Y_val_df, epochs=15, batch_size=100, verbose=True)

# Evaluate on validation data
val_accuracy_df = model_df.evaluate_accuracy(X_val_df, Y_val_df)
print(f"Validation Accuracy (DF): {val_accuracy_df:.2f}%")

# Predict soft values for RMSE calculation on training set
soft_predictions_train_df = model_df.predict_soft(X_train_df)

# Calculate the true labels from one-hot encoding (convert one-hot back to integer values)
true_labels_train_df = np.argmax(Y_train_df, axis=1) + 1  # Add 1 to match 1-5 stars

# Calculate RMSE on DF training data
rmse_df = np.sqrt(mean_squared_error(true_labels_train_df, soft_predictions_train_df))
print(f"RMSE (DF Training Set): {rmse_df:.4f}")


# --------------- Train and Evaluate CTF Model -----------------

# Train the RMLR model using CTF features
num_features_ctf = features_ctf_train.shape[1]

model_ctf = rmlr123(num_features=num_features_ctf, num_classes=num_classes,
                    learning_rate=learning_rate, lambda_reg=lambda_reg,
                    class_weights=class_weights)

X_train_ctf, X_val_ctf, Y_train_ctf, Y_val_ctf = train_test_split(features_ctf_train, y_one_hot, test_size=0.2, random_state=42)

# Train the model
model_ctf.train(X_train_ctf, Y_train_ctf, X_val=X_val_ctf, Y_val=Y_val_ctf, epochs=10, batch_size=100, verbose=True)

# Evaluate on validation data
val_accuracy_ctf = model_ctf.evaluate_accuracy(X_val_ctf, Y_val_ctf)
print(f"Validation Accuracy (CTF): {val_accuracy_ctf:.2f}%")

# Predict soft values for RMSE calculation on training set
soft_predictions_train_ctf = model_ctf.predict_soft(X_train_ctf)

# Calculate the true labels from one-hot encoding (convert one-hot back to integer values)
true_labels_train_ctf = np.argmax(Y_train_ctf, axis=1) + 1  # Add 1 to match 1-5 stars

# Calculate RMSE on CTF training data
rmse_ctf = np.sqrt(mean_squared_error(true_labels_train_ctf, soft_predictions_train_ctf))
print(f"RMSE (CTF Training Set): {rmse_ctf:.4f}")

# %%
def generate_predictions(model, X, output_file):
    """
    Generate hard and soft predictions for a dataset and write them to a file.

    Args:
        model (rmlr123): Trained RMLR model.
        X (np.ndarray): Feature matrix for the dataset (dev or test set).
        output_file (str): Path to the output file to write predictions.
    """
    hard_predictions = model.predict_hard(X)  # Hard predictions
    soft_predictions = model.predict_soft(X)  # Soft predictions

    with open(output_file, 'w') as f:
        for hard_pred, soft_pred in zip(hard_predictions, soft_predictions):
            f.write(f"{hard_pred} {soft_pred:.4f}\n")  # Write hard and soft prediction

X_dev = features_ctf_dev  # Replace with your actual dev set features
X_test = features_ctf_test  # Replace with your actual test set features

# Generate predictions for development and test sets
#generate_predictions(model_ctf, X_dev, "dev-predictions-ctf.txt")
#generate_predictions(model_ctf, X_test, "test-predictions-ctf.txt")

X_dev = features_df_dev  # Replace with your actual dev set features
X_test = features_df_test  # Replace with your actual test set features

# Generate predictions for development and test sets
#generate_predictions(model_df, X_dev, "dev-predictions-df.txt")
#generate_predictions(model_df, X_test, "test-predictions-df.txt")

print("Prediction files generated successfully.")


