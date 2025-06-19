from datasets import load_dataset, get_dataset_split_names
import random
from train_cbow.cbow_utils import load_vocab, cbow_tokenize
import torch
import torch.nn as nn
from train_cbow.cbow_model import CBOW

def get_marco_ds_splits():
    # get_dataset_split_names("microsoft/ms_marco", "v1.1")
    dataset = load_dataset("microsoft/ms_marco", "v1.1", cache_dir="data/raw/marco")
    return dataset["train"], dataset["validation"], dataset["test"] # type: ignore	


def generate_triplets(dataset_split, max_negatives_per_query=1):
    triplets = []

    for sample in dataset_split:
        query = sample["query"]
        passages = sample["passages"]["passage_text"]
        labels = sample["passages"]["is_selected"]

        positives = [p for p, sel in zip(passages, labels) if sel == 1]
        negatives = [p for p, sel in zip(passages, labels) if sel == 0]

        if not positives or not negatives:
            continue  # Skip if no usable data

        for pos in positives:
            num_negs = min(max_negatives_per_query, len(negatives))
            unique_negs = random.sample(negatives, num_negs)
            for neg in unique_negs:
                triplets.append((query, pos, neg))
    return triplets

def generate_triplet_tokens(triplets, vocab_dict=None):
    triplet_tokens = []
    for query, pos, neg in triplets:
        query_tokens = cbow_tokenize(query, vocab_dict)
        pos_tokens = cbow_tokenize(pos, vocab_dict)
        neg_tokens = cbow_tokenize(neg, vocab_dict)
        triplet_tokens.append((query_tokens, pos_tokens, neg_tokens))

    return triplet_tokens

class MARCOTripletDataset(torch.utils.data.Dataset):
    def __init__(self, triplets, vocab_path=None, vocab_size=None):
        self.triplets = triplets
        self.vocab_dict = load_vocab(vocab_path=vocab_path, vocab_size=vocab_size)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        query, pos, neg = self.triplets[idx]
        query_tokens = cbow_tokenize(query, self.vocab_dict)
        pos_tokens = cbow_tokenize(pos, self.vocab_dict)
        neg_tokens = cbow_tokenize(neg, self.vocab_dict)
        return query_tokens, pos_tokens, neg_tokens

def load_frozen_embedding_from_cbow(checkpoint_path, vocab_size, emb_dim):
    cbow = CBOW(vocab_size, emb_dim)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    # If checkpoint is a dict with 'model_state_dict', extract it
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    cbow.load_state_dict(state_dict)
    embedding = nn.Embedding(vocab_size, emb_dim)
    embedding.weight.data.copy_(cbow.embeddings.weight.data)
    embedding.weight.requires_grad = False  # Freeze weights
    return embedding

if __name__ == "__main__":
    train_split, val_split, test_split = get_marco_ds_splits()
    triplets = generate_triplets(train_split, max_negatives_per_query=1)
    print(f"Generated {len(triplets)} triplets from the training split.")
    tokens = generate_triplet_tokens(triplets, vocab_path='train_cbow/tkn_words_to_ids.pkl', vocab_size=30000)
    for index in random.sample(range(len(triplets)), 5):
        print(f"Triplet {index}: {triplets[index]}\n")
        print(f"Tokens: {tokens[index]}\n")
    # You can save or process the triplets further as needed.
