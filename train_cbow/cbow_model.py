import torch
import torch.nn as nn


class CBOW(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.linear = nn.Linear(in_features=emb_dim, out_features=vocab_size, bias=False)

    def forward(self, inpt):
        embeds = self.embeddings(inpt) # shape: (batch, context_size, emb_dim)
        embeds = embeds.mean(dim=1) # shape: (batch, emb_dim)
        out = self.linear(embeds) # shape: (batch, vocab_size)
        return out