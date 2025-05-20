import torch
import torch.nn as nn

class CNNClassifier(nn.Module):
    """
    CNN-based text classifier.
    """
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, output_dim, padding_idx, pretrained_embeddings=None, freeze_embeddings=False):
        """
        Initializes the CNN classifier.
        
        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of word embeddings.
            num_filters (int): Number of filters per filter size.
            filter_sizes (list): List of filter sizes (kernel widths).
            output_dim (int): Number of output classes.
            padding_idx (int): Index used for padding tokens.
            pretrained_embeddings (torch.Tensor, optional): Pre-trained embedding weights.
            freeze_embeddings (bool, optional): Whether to freeze embedding weights.
        """
        super(CNNClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = not freeze_embeddings

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=num_filters,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)

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
        embedded = embedded.permute(0, 2, 1)  

        conv_outputs = [torch.relu(conv(embedded)) for conv in self.convs]  

        pooled_outputs = [torch.max(conv_out, dim=2)[0] for conv_out in conv_outputs]  

        cat = torch.cat(pooled_outputs, dim=1) 

        cat = self.dropout(cat)

        out = self.fc(cat)
        return out