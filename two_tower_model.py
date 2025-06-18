import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTowerModel(nn.Module):
    """
    Two Tower architecture for retrieval tasks.
    Uses a shared embedding layer and two separate RNN encoders
    (one for queries, one for documents).
    """
    def __init__(self, embedding_layer, embedding_dim, rnn_hidden_dim, num_rnn_layers=1, rnn_type='lstm'):
        """
        Args:
            embedding_layer (nn.Embedding): Pretrained embedding layer (shared).
            embedding_dim (int): Dimension of word embeddings.
            rnn_hidden_dim (int): Hidden state size for RNNs.
            num_rnn_layers (int): Number of RNN layers.
            rnn_type (str): 'lstm' or 'gru'.
        """
        super().__init__()
        self.embedding = embedding_layer
        self.embedding_dim = embedding_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_rnn_layers = num_rnn_layers

        # Choose RNN type
        if rnn_type.lower() == 'lstm':
            rnn_class = nn.LSTM
        elif rnn_type.lower() == 'gru':
            rnn_class = nn.GRU
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

        # Separate encoders for queries and documents
        self.query_encoder = rnn_class(
            input_size=embedding_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=num_rnn_layers,
            batch_first=True
        )
        self.document_encoder = rnn_class(
            input_size=embedding_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=num_rnn_layers,
            batch_first=True
        )

    def encode_query(self, query_token_ids):
        """
        Encodes a batch of queries.
        Args:
            query_token_ids (Tensor): (batch_size, query_seq_len)
        Returns:
            Tensor: (batch_size, rnn_hidden_dim)
        """
        embedded_queries = self.embedding(query_token_ids)
        rnn_output, hidden_state = self.query_encoder(embedded_queries)
        # For LSTM, hidden_state is a tuple (h_n, c_n)
        if isinstance(hidden_state, tuple):
            hidden_state = hidden_state[0]
        # Use the last layer's hidden state
        return hidden_state[-1]

    def encode_document(self, document_token_ids):
        """
        Encodes a batch of documents.
        Args:
            document_token_ids (Tensor): (batch_size, doc_seq_len)
        Returns:
            Tensor: (batch_size, rnn_hidden_dim)
        """
        embedded_documents = self.embedding(document_token_ids)
        rnn_output, hidden_state = self.document_encoder(embedded_documents)
        if isinstance(hidden_state, tuple):
            hidden_state = hidden_state[0]
        return hidden_state[-1]

    def forward(self, query_token_ids, positive_doc_token_ids, negative_doc_token_ids):
        """
        Forward pass for a batch of triplets.
        Args:
            query_token_ids (Tensor): (batch_size, query_seq_len)
            positive_doc_token_ids (Tensor): (batch_size, doc_seq_len)
            negative_doc_token_ids (Tensor): (batch_size, doc_seq_len)
        Returns:
            Tuple[Tensor, Tensor, Tensor]: Encoded query, positive doc, negative doc vectors.
        """
        query_vector = self.encode_query(query_token_ids)
        positive_doc_vector = self.encode_document(positive_doc_token_ids)
        negative_doc_vector = self.encode_document(negative_doc_token_ids)
        return query_vector, positive_doc_vector, negative_doc_vector 