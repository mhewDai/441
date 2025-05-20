import torch
import numpy as np
import gc
import pandas as pd
from tqdm.auto import tqdm
import os
import datetime
from collections import Counter
import re
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import seaborn as sns
import torch.optim as optim
from CNN import CNNClassifier
from LSTM import LSTMClassifier

def prepare_embedding_matrix(vocab_size, embedding_dim, vocabulary, pretrained_embeddings, use_pretrained):
    """
    Prepares the embedding matrix either with pre-trained embeddings or random initialization.
    
    Args:
        vocab_size (int): size of the vocabulary.
        embedding_dim (int): Dimension of embeddings.
        vocabulary (dict): Mapping from words to indices.
        pretrained_embeddings (dict): Pre-trained word embeddings.
        use_pretrained (bool): Whether to use pre-trained embeddings.
    
    Returns:
        torch.Tensor: Embedding matrix.
    """
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, idx in vocabulary.items():
        if use_pretrained and word in pretrained_embeddings:
            embedding_matrix[idx] = pretrained_embeddings[word]
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim, ))

    return torch.tensor(embedding_matrix, dtype=torch.float)

def tokenize(text):
    """
    Tokenizes the input text into lowercase words.
    
    Args:
        text (str): The text to tokenize.
    
    Returns:
        list: A list of tokens.
    """
    tokens = re.findall(r'\S+', text.lower())
    return tokens

def extract_tokens_and_lengths(folder_path):
    """
    Extracts tokens and their lengths from all text files in a given folder.
    
    Args:
        folder_path (str): Path to the folder containing text files.
    
    Returns:
        tuple: A tuple containing:
            - tokens_per_doc (list of lists): Tokens from each document.
            - document_lengths (list): Number of tokens in each document.
    """
    tokens_per_doc = []
    document_lengths = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            file_tokens = tokenize(text)
            tokens_per_doc.append(file_tokens)
            document_lengths.append(len(file_tokens))
    return tokens_per_doc, document_lengths

def text_to_sequence(tokens, vocabulary):
    """
    Converts a list of tokens into a list of corresponding indices based on the vocabulary.
    
    Args:
        tokens (list): List of tokens.
        vocabulary (dict): Mapping from tokens to indices.
    
    Returns:
        list: List of token indices.
    """
    return [vocabulary.get(token, vocabulary['<UNK>']) for token in tokens]

def pad_truncate(sequence, max_length, padding_value=0):
    """
    Pads or truncates a sequence to a fixed length.
    
    Args:
        sequence (list): List of token indices.
        max_length (int): Desired sequence length.
        padding_value (int, optional): Value to use for padding. Defaults to 0.
    
    Returns:
        list: Padded or truncated sequence.
    """
    if len(sequence) > max_length:
        return sequence[:max_length]
    else:
        return sequence + [padding_value] * (max_length - len(sequence))

def load_pretrained_embeddings(embedding_file):
    """
    Loads pre-trained word embeddings from a file.
    
    Args:
        embedding_file (str): Path to the embedding file.
    
    Returns:
        dict: Mapping from words to their embedding vectors.
    """
    embeddings = {}
    with open(embedding_file, 'r', encoding='utf-8') as f:
        first_line = f.readline()  # Skip the first line if necessary
        for line in f:
            values = line.split()
            if len(values) < 2:
                continue  # Skip invalid lines
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def calculate_accuracy(preds, y):
    """
    Calculates the accuracy of predictions.
    
    Args:
        preds (torch.Tensor): Logits from the model (batch_size, output_dim)
        y (torch.Tensor): True labels (batch_size)
    
    Returns:
        float: Accuracy as a percentage
    """
    _, predicted = torch.max(preds, 1)
    correct = (predicted == y).float()
    acc = correct.sum() / len(correct)
    return acc * 100

def train_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    
    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (nn.Module): Loss function.
        device (str): Device to run the model on.
    
    Returns:
        tuple: Average loss and accuracy for the epoch.
    """
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    progress_bar = tqdm(loader, desc="Training", leave=False)

    for sequences, labels in progress_bar:
        sequences = sequences.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        acc = calculate_accuracy(outputs, labels)
        epoch_acc += acc

        # Update the progress bar with current loss and accuracy
        progress_bar.set_postfix({'loss': loss.item(), 'accuracy': acc})

    return epoch_loss / len(loader), epoch_acc / len(loader)

def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on validation or test data.
    
    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): Validation or Test DataLoader.
        criterion (nn.Module): Loss function.
        device (str): Device to run the model on.
    
    Returns:
        tuple: Average loss and accuracy.
    """
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            outputs = model(sequences)
            loss = criterion(outputs, labels)

            epoch_loss += loss.item()
            acc = calculate_accuracy(outputs, labels)
            epoch_acc += acc

    return epoch_loss / len(loader), epoch_acc / len(loader)

def epoch_time(start_time, end_time):
    """
    Calculates the elapsed time between two time points.
    
    Args:
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.
    
    Returns:
        tuple: Elapsed minutes and seconds.
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

class TextDataset(Dataset):
    """
    Custom Dataset for text data.
    """
    def __init__(self, sequences, labels):
        """
        Initializes the dataset with sequences and labels.
        
        Args:
            sequences (list of lists): Padded/truncated sequences of word indices.
            labels (list): Corresponding labels for each sequence.
        """
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]
    
def filter_descriptions(model_type, accuracy_histories):
    """
    Filters the accuracy_histories dictionary for a specific model type.

    Args:
        model_type (str): The type of the model ('LSTM' or 'CNN').
        accuracy_histories (dict): Dictionary containing accuracy histories.

    Returns:
        dict: Filtered dictionary containing only the specified model type.
    """
    filtered = {desc: acc for desc, acc in accuracy_histories.items() if desc.startswith(model_type)}
    return filtered

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)

    #The locations of each dataset folders
    train_root = 'data/train'
    test_root = 'data/test'

    train_positive_path = os.path.join(train_root, 'positive')
    train_negative_path = os.path.join(train_root, 'negative')

    #Extracting tokens
    print("Extracting training data...")
    train_positive_tokens, train_positive_lengths = extract_tokens_and_lengths(train_positive_path)
    train_negative_tokens, train_negative_lengths = extract_tokens_and_lengths(train_negative_path)

    #Calculating stats for part1
    all_training_tokens = [token for doc in train_positive_tokens + train_negative_tokens for token in doc]
    all_training_lengths = train_positive_lengths + train_negative_lengths

    word_counter = Counter(all_training_tokens)
    total_unique_words = len(word_counter)
    total_training_examples = len(train_positive_lengths) + len(train_negative_lengths)
    num_positive = len(train_positive_lengths)
    num_negative = len(train_negative_lengths)
    ratio_positive_to_negative = num_positive / num_negative if num_negative != 0 else float('inf')
    average_document_length = sum(all_training_lengths) / len(all_training_lengths)
    max_document_length = max(all_training_lengths)

    #My stats
    print("\nTraining Set Statistics:")
    print(f"Total number of unique words: {total_unique_words}")
    print(f"Total number of training examples: {total_training_examples}")
    print(f"Ratio of positive to negative examples: {ratio_positive_to_negative:.2f}")
    print(f"Average document length: {average_document_length:.2f}")
    print(f"Max document length: {max_document_length}")

    #Creating vocab
    vocab_size = 10000
    most_common_words = word_counter.most_common(vocab_size - 2)

    vocabulary = {'<PAD>': 0, '<UNK>': 1}
    for idx, (word, count) in enumerate(most_common_words, start=2):
        vocabulary[word] = idx

    train_positive_sequences = [text_to_sequence(tokens, vocabulary) for tokens in train_positive_tokens]
    train_negative_sequences = [text_to_sequence(tokens, vocabulary) for tokens in train_negative_tokens]

    all_training_sequences = train_positive_sequences + train_negative_sequences

    embedding_file_path = 'data/all.review.vec.txt'

    print("\nLoading pre-trained embeddings...")
    pretrained_embeddings = load_pretrained_embeddings(embedding_file_path)

    embedding_dim = 100
    lmax = 100

    labels = [1] * num_positive + [0] * num_negative

    print("\nPadding/truncating training sequences...")
    train_sequences_padded = [pad_truncate(seq, lmax, padding_value=vocabulary['<PAD>']) for seq in all_training_sequences]

    print("Splitting data into training and validation sets...")
    train_seqs, val_seqs, train_labels, val_labels = train_test_split(
        train_sequences_padded,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    test_positive_path = os.path.join(test_root, 'positive')
    test_negative_path = os.path.join(test_root, 'negative')

    print("\nExtracting test data...")
    test_positive_tokens, test_positive_lengths = extract_tokens_and_lengths(test_positive_path)
    test_negative_tokens, test_negative_lengths = extract_tokens_and_lengths(test_negative_path)

    test_positive_sequences = [text_to_sequence(tokens, vocabulary) for tokens in test_positive_tokens]
    test_negative_sequences = [text_to_sequence(tokens, vocabulary) for tokens in test_negative_tokens]

    all_test_sequences = test_positive_sequences + test_negative_sequences

    print("Padding/truncating test sequences...")
    test_sequences_padded = [pad_truncate(seq, lmax, padding_value=vocabulary['<PAD>']) for seq in all_test_sequences]

    test_labels = [1] * len(test_positive_sequences) + [0] * len(test_negative_sequences)

    batch_size = 64

    #Data Loaders
    train_dataset = TextDataset(train_seqs, train_labels)
    val_dataset = TextDataset(val_seqs, val_labels)
    test_dataset = TextDataset(test_sequences_padded, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"\nTrain dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    hidden_dim = 100
    output_dim = 2
    padding_idx = vocabulary['<PAD>']

    vocab_size = max(vocabulary.values()) + 1
    print(f"Adjusted vocab_size: {vocab_size}")

    #Loss Function
    criterion = nn.CrossEntropyLoss()

    #Model Types
    model_types = ['LSTM', 'CNN']

    #Pretrained/Not pretrained
    use_pretrained_options = [False, True]

    #CNN parameters
    cnn_params = {
        'num_filters': 100,
        'filter_sizes': [3, 4, 5]
    }

    lstm_params = {
        'hidden_dim': 100
    }

    # Dictionary to store results
    results = {
        'description': [],
        'training_time': [],
        'train_accuracy_history': [],   
        'validation_accuracy_history': [],  
        'test_accuracy_history': []
    }

    accuracy_histories = {}

    for model_type in model_types:
        for use_pretrained in use_pretrained_options:
            description = f"{model_type} {'with' if use_pretrained else 'without'} Pre-trained Embeddings"
            print(f"\n{'='*60}")
            print(f"Training Configuration: {description}")
            print(f"{'='*60}")

            embedding_matrix = prepare_embedding_matrix(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                vocabulary=vocabulary,
                pretrained_embeddings=pretrained_embeddings,
                use_pretrained=use_pretrained
            )

            if use_pretrained:
                pretrained_embedding = embedding_matrix
            else:
                pretrained_embedding = None

            if model_type == "LSTM":
                model = LSTMClassifier(
                    vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                    hidden_dim=lstm_params['hidden_dim'],
                    output_dim=output_dim,
                    padding_idx=padding_idx,
                    pretrained_embeddings=pretrained_embedding,
                    freeze_embeddings=False
                ).to(device)
            elif model_type == "CNN":
                model = CNNClassifier(
                    vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                    num_filters=cnn_params['num_filters'],
                    filter_sizes=cnn_params['filter_sizes'],
                    output_dim=output_dim,
                    padding_idx=padding_idx,
                    pretrained_embeddings=pretrained_embedding,
                    freeze_embeddings=False
                ).to(device)
            else:
                raise ValueError("Invalid model type selected!")

            optimizer = optim.Adam(model.parameters(), lr=0.001)

            NUM_EPOCHS = 10
            start_time_total = time.time()

            train_acc_history = []
            val_acc_history = []
            test_acc_history = []

            for epoch in range(1, NUM_EPOCHS + 1):
                start_time = time.time()

                train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                test_loss, test_acc = evaluate(model, test_loader, criterion, device)

                train_acc_history.append(train_acc.item())
                val_acc_history.append(val_acc.item())
                test_acc_history.append(test_acc.item())

                epoch_duration = time.time() - start_time
                print(f"Epoch {epoch}/{NUM_EPOCHS} | "
                      f"Train Acc: {train_acc.item():.2f}% | "
                      f"Val Acc: {val_acc.item():.2f}% | "
                      f"Test Acc: {test_acc.item():.2f}% | "
                      f"Time: {epoch_duration:.2f}s")

            # Store the final results for this configuration
            results['description'].append(description)
            end_time_total = time.time()
            epoch_time_total = end_time_total - start_time_total
            results['training_time'].append(epoch_time_total)
            results['train_accuracy_history'].append(train_acc_history)       # Store full history
            results['validation_accuracy_history'].append(val_acc_history)
            results['test_accuracy_history'].append(test_acc_history)

            # Also store in accuracy_histories for easier plotting
            accuracy_histories[description] = {
                'train': train_acc_history,
                'val': val_acc_history,
                'test': test_acc_history
            }

    # After training all configurations, print the final results
    results_df = pd.DataFrame({
        'Description': results['description'],
        'Training Time (s)': results['training_time'],
        'Final Train Accuracy (%)': [acc[-1] for acc in results['train_accuracy_history']],
        'Final Validation Accuracy (%)': [acc[-1] for acc in results['validation_accuracy_history']],
        'Final Test Accuracy (%)': [acc[-1] for acc in results['test_accuracy_history']]
    })
    print("\nFinal Results:")
    print(results_df)

    results_df.to_csv('results.csv', index=False)
    print("\nFinal results saved to 'results.csv'.")


    plt.figure(figsize=(14, 8))
    filtered_lstm_histories = filter_descriptions('LSTM', accuracy_histories)

    for description, accuracies in filtered_lstm_histories.items():
        epochs = list(range(1, NUM_EPOCHS + 1))
        plt.plot(epochs, accuracies['train'], label=f'{description} Train')
        plt.plot(epochs, accuracies['test'], label=f'{description} Test')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Testing Accuracy Over Time for All LSTM Configurations')
    plt.xticks(epochs)
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plotting Accuracy Over Time for All CNN Configurations
    plt.figure(figsize=(14, 8))
    filtered_cnn_histories = filter_descriptions('CNN', accuracy_histories)

    for description, accuracies in filtered_cnn_histories.items():
        epochs = list(range(1, NUM_EPOCHS + 1))
        plt.plot(epochs, accuracies['train'], label=f'{description} Train')
        plt.plot(epochs, accuracies['test'], label=f'{description} Test')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Testing Accuracy Over Time for All CNN Configurations')
    plt.xticks(epochs)
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.show()
