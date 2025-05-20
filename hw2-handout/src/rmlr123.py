import numpy as np

class rmlr123:
    def __init__(self, num_features, num_classes, learning_rate=0.01, lambda_reg=0.1, class_weights=None):
        """
        Initializes the RMLR model parameters.

        Args:
            num_features (int): Number of features (D).
            num_classes (int): Number of classes (C).
            learning_rate (float): Learning rate for SGD.
            lambda_reg (float): Regularization parameter.
            class_weights (np.ndarray, optional): Class weights for handling imbalanced classes. Should be an array of shape (num_classes,).
        """
        self.num_features = num_features
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.W = np.random.randn(num_classes, num_features) * 0.01
        self.class_weights = class_weights if class_weights is not None else np.ones(num_classes)

    def softmax(self, scores):
        scores_shift = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores_shift)
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probabilities

    def compute_loss(self, X, Y):
        scores = np.dot(X, self.W.T)
        probabilities = self.softmax(scores)
        epsilon = 1e-15
        log_probs = np.log(probabilities + epsilon)
        
        # Reshape class_weights for broadcasting
        class_weights = self.class_weights.reshape(1, -1)
        weighted_log_probs = Y * log_probs * class_weights
        loss = -np.sum(weighted_log_probs) / X.shape[0]

        # L2 Regularization
        loss += (self.lambda_reg / 2) * np.sum(self.W * self.W)
        return loss

    def compute_gradient(self, X, Y):
        scores = np.dot(X, self.W.T)
        probabilities = self.softmax(scores)
        diff = probabilities - Y  # Shape: (n_samples, num_classes)
        
        # Apply class weights
        class_weights = self.class_weights.reshape(1, -1)
        weighted_diff = diff * class_weights  # Broadcasting over axis 0
        
        # Compute gradient
        gradient = np.dot(weighted_diff.T, X) / X.shape[0]
        
        # Regularization term for gradient
        gradient += self.lambda_reg * self.W
        return gradient

    def train(self, X_train, Y_train, X_val=None, Y_val=None, epochs=10, batch_size=100, verbose=True):
        n_samples = X_train.shape[0]
        for epoch in range(1, epochs + 1):
            permutation = np.random.permutation(n_samples)
            X_shuffled = X_train[permutation]
            Y_shuffled = Y_train[permutation]

            epoch_loss = 0.0
            correct = 0

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                Y_batch = Y_shuffled[i:i + batch_size]

                # Compute gradient and loss for the mini-batch
                loss = self.compute_loss(X_batch, Y_batch)
                gradient = self.compute_gradient(X_batch, Y_batch)

                # Update weights using the mini-batch gradient
                self.W -= self.learning_rate * gradient

                # Accumulate epoch loss
                epoch_loss += loss * X_batch.shape[0]

                # Calculate accuracy on the mini-batch for tracking purposes
                scores = np.dot(X_batch, self.W.T)
                probabilities = self.softmax(scores)
                predictions = np.argmax(probabilities, axis=1)
                true_labels = np.argmax(Y_batch, axis=1)
                correct += np.sum(predictions == true_labels)

            # Average loss over the epoch
            avg_loss = epoch_loss / n_samples
            accuracy = (correct / n_samples) * 100

            if verbose:
                print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")

                # Validation performance if validation set is provided
                if X_val is not None and Y_val is not None:
                    val_loss = self.compute_loss(X_val, Y_val)
                    val_accuracy = self.evaluate_accuracy(X_val, Y_val)
                    print(f"    Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_accuracy:.2f}%")

    def predict_hard(self, X):
        scores = np.dot(X, self.W.T)
        probabilities = self.softmax(scores)
        predictions = np.argmax(probabilities, axis=1) + 1  # Adding 1 to match class labels 1-5
        return predictions

    def predict_soft(self, X):
        scores = np.dot(X, self.W.T)
        probabilities = self.softmax(scores)
        class_indices = np.arange(1, self.num_classes + 1)
        soft_predictions = np.dot(probabilities, class_indices)
        return soft_predictions

    def evaluate_accuracy(self, X, Y):
        predictions = self.predict_hard(X)
        true_labels = np.argmax(Y, axis=1) + 1  # Adding 1 to match class labels 1-5
        accuracy = (np.sum(predictions == true_labels) / X.shape[0]) * 100
        return accuracy
