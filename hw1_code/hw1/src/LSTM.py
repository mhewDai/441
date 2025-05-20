import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    """
    LSTM-based text classifier.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, padding_idx, pretrained_embeddings=None, freeze_embeddings=False):
        """
        Initializes the LSTM classifier.
        
        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of word embeddings.
            hidden_dim (int): Dimension of LSTM hidden states.
            output_dim (int): Number of output classes.
            padding_idx (int): Index used for padding tokens.
            pretrained_embeddings (torch.Tensor, optional): Pre-trained embedding weights.
            freeze_embeddings (bool, optional): Whether to freeze embedding weights.
        """
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = not freeze_embeddings  

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)  

        self.dropout = nn.Dropout(0.5) 

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length)

        Returns:
            torch.Tensor: Output logits of shape (batch_size, output_dim)
        """
        embedded = self.embedding(x)            
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        hidden = self.dropout(hidden)  
        out = self.fc(hidden)               
        return out

