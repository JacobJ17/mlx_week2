import torch
import torch.nn as nn
import numpy as np

def load_embedding_layer(embedding_type, vocab, emb_dim, embedding_path=None, freeze=True, cbow_vocab_size=None):
    """
    Load embedding layer from different sources.
    
    Args:
        embedding_type (str): 'cbow', 'glove', or 'random'
        vocab (dict): Dictionary mapping word to index
        emb_dim (int): Embedding dimension
        embedding_path (str): Path to embedding file (if needed)
        freeze (bool): If True, embeddings are not trainable
    
    Returns:
        nn.Embedding: Embedding layer with loaded weights
    """
    vocab_size = len(vocab) if embedding_type != 'cbow' else cbow_vocab_size
    embedding = nn.Embedding(vocab_size, emb_dim)
    
    if embedding_type == 'cbow':
        # Load from your CBOW checkpoint
        from two_tower_utils import load_frozen_embedding_from_cbow
        embedding = load_frozen_embedding_from_cbow(embedding_path, vocab_size, emb_dim)
        embedding.weight.requires_grad = not freeze
        return embedding

    elif embedding_type == 'glove':
        # Load GloVe vectors from .txt file
        if embedding_path is None:
            raise ValueError("embedding_path must be provided for GloVe embeddings")
            
        weights = np.random.randn(vocab_size, emb_dim) * 0.01  # fallback for OOV
        
        # Define special tokens that exist in CBOW vocab but not in GloVe
        special_tokens = [
            '<DELIMIT>', '<HYPHEN>', '<PERIOD>', '<COMMA>', 
            '<SEMICOLON>', '<EXCLAMATION_MARK>', '<QUESTION_MARK>'
        ]
        
        # Load GloVe file
        glove = {}
        with open(embedding_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                vec = np.array(parts[1:], dtype=np.float32)
                glove[word] = vec
        
        # Map vocabulary to embeddings
        for word, idx in vocab.items():
            if word in glove:
                # Regular word found in GloVe
                weights[idx] = glove[word]
            elif word in special_tokens:
                # Special token from CBOW vocab - use small random initialization
                weights[idx] = np.random.randn(emb_dim) * 0.1
            # Other OOV words get the default small random initialization (0.01)
        
        embedding.weight.data.copy_(torch.tensor(weights, dtype=torch.float))
        embedding.weight.requires_grad = not freeze
        return embedding

    elif embedding_type == 'random':
        # Just use random initialization
        embedding.weight.requires_grad = not freeze
        return embedding

    else:
        raise ValueError(f'Unknown embedding_type: {embedding_type}') 