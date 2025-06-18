import torch
import torch.nn as nn
from train_cbow.cbow_model import CBOW

def load_frozen_embedding_from_cbow(checkpoint_path, vocab_size, emb_dim):
    """
    Loads embedding weights from a CBOW checkpoint and returns a frozen nn.Embedding layer.
    Args:
        checkpoint_path (str): Path to the CBOW .pth checkpoint.
        vocab_size (int): Number of tokens in the vocabulary.
        emb_dim (int): Embedding dimension.
    Returns:
        nn.Embedding: Embedding layer with pretrained weights, frozen.
    """
    cbow = CBOW(vocab_size, emb_dim)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    # If checkpoint is a dict with 'model_state_dict', extract it
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    cbow.load_state_dict(state_dict)
    embedding = nn.Embedding(vocab_size, emb_dim)
    embedding.weight.data.copy_(cbow.embeddings.weight.data)
    embedding.weight.requires_grad = False  # Freeze weights
    return embedding 