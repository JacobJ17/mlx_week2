import torch
import torch.nn as nn
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader
from two_tower_utils import load_frozen_embedding_from_cbow
from two_tower_model import TwoTowerModel
from data_utils import get_marco_ds_splits, generate_triplets, MARCOTripletDataset
from embedding_loader import load_embedding_layer
from train_cbow.cbow_utils import load_vocab
import wandb
import os
import argparse
from tqdm import tqdm

torch.manual_seed(42)
torch.cuda.manual_seed(42)

def collate_fn(batch):
    """
    Collate function for batching triplets with padding and lengths.
    batch: list of (query_tokens, pos_tokens, neg_tokens)
    Return: queries_padded, queries_lens, positives_padded, positives_lens, negatives_padded, negatives_lens
    """
    queries, positives, negatives = zip(*batch)
    queries = [torch.tensor(q, dtype=torch.long) for q in queries]
    positives = [torch.tensor(p, dtype=torch.long) for p in positives]
    negatives = [torch.tensor(n, dtype=torch.long) for n in negatives]
    queries_lens = torch.tensor([len(q) for q in queries], dtype=torch.long)
    positives_lens = torch.tensor([len(p) for p in positives], dtype=torch.long)
    negatives_lens = torch.tensor([len(n) for n in negatives], dtype=torch.long)
    queries_padded = pad_sequence(queries, batch_first=True, padding_value=0)
    positives_padded = pad_sequence(positives, batch_first=True, padding_value=0)
    negatives_padded = pad_sequence(negatives, batch_first=True, padding_value=0)
    return queries_padded, queries_lens, positives_padded, positives_lens, negatives_padded, negatives_lens

def encode_with_packing(embedding, rnn, token_ids, lengths):
    # Sort by length (descending)
    lengths, perm_idx = lengths.sort(0, descending=True)
    token_ids = token_ids[perm_idx]
    embedded = embedding(token_ids)
    packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=True)
    output, hidden = rnn(packed)
    if isinstance(hidden, tuple):  # LSTM
        hidden = hidden[0]
    # Undo the sorting
    _, unperm_idx = perm_idx.sort(0)
    return hidden[-1][unperm_idx]

def train_one_epoch(model, dataloader, optimizer, device, margin=0.2):
    model.train()
    total_loss = 0
    num_batches = 0
    for batch in tqdm(dataloader, desc='Training', leave=False):
        query_tokens, query_lens, pos_tokens, pos_lens, neg_tokens, neg_lens = batch
        query_tokens = query_tokens.to(device)
        query_lens = query_lens.to(device)
        pos_tokens = pos_tokens.to(device)
        pos_lens = pos_lens.to(device)
        neg_tokens = neg_tokens.to(device)
        neg_lens = neg_lens.to(device)
        # Use packed sequence encoding
        query_vec = encode_with_packing(model.embedding, model.query_encoder, query_tokens, query_lens)
        pos_vec = encode_with_packing(model.embedding, model.document_encoder, pos_tokens, pos_lens)
        neg_vec = encode_with_packing(model.embedding, model.document_encoder, neg_tokens, neg_lens)
        pos_sim = torch.nn.functional.cosine_similarity(query_vec, pos_vec)
        neg_sim = torch.nn.functional.cosine_similarity(query_vec, neg_vec)
        triplet_loss = torch.nn.functional.relu(neg_sim - pos_sim + margin).mean()
        optimizer.zero_grad()
        triplet_loss.backward()
        optimizer.step()
        total_loss += triplet_loss.item()
        num_batches += 1
    return total_loss / num_batches

def validate_one_epoch(model, dataloader, device, margin=0.2):
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating', leave=False):
            query_tokens, query_lens, pos_tokens, pos_lens, neg_tokens, neg_lens = batch
            query_tokens = query_tokens.to(device)
            query_lens = query_lens.to(device)
            pos_tokens = pos_tokens.to(device)
            pos_lens = pos_lens.to(device)
            neg_tokens = neg_tokens.to(device)
            neg_lens = neg_lens.to(device)
            query_vec = encode_with_packing(model.embedding, model.query_encoder, query_tokens, query_lens)
            pos_vec = encode_with_packing(model.embedding, model.document_encoder, pos_tokens, pos_lens)
            neg_vec = encode_with_packing(model.embedding, model.document_encoder, neg_tokens, neg_lens)
            pos_sim = torch.nn.functional.cosine_similarity(query_vec, pos_vec)
            neg_sim = torch.nn.functional.cosine_similarity(query_vec, neg_vec)
            triplet_loss = torch.nn.functional.relu(neg_sim - pos_sim + margin).mean()
            total_loss += triplet_loss.item()
            num_batches += 1
    return total_loss / num_batches

def recall_at_k(model, dataloader, device, k=1):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Recall@{k}', leave=False):
            query_tokens, query_lens, pos_tokens, pos_lens, neg_tokens, neg_lens = batch
            query_tokens = query_tokens.to(device)
            query_lens = query_lens.to(device)
            pos_tokens = pos_tokens.to(device)
            pos_lens = pos_lens.to(device)
            neg_tokens = neg_tokens.to(device)
            neg_lens = neg_lens.to(device)
            batch_size = query_tokens.size(0)

            # Encode queries and all docs (positive and negative)
            query_vecs = encode_with_packing(model.embedding, model.query_encoder, query_tokens, query_lens)  # (batch, hidden_dim)
            pos_vecs = encode_with_packing(model.embedding, model.document_encoder, pos_tokens, pos_lens)   # (batch, hidden_dim)
            neg_vecs = encode_with_packing(model.embedding, model.document_encoder, neg_tokens, neg_lens)   # (batch, hidden_dim)

            # For each query, create a candidate set: [pos_doc, neg_doc]
            # (You can extend this to more negatives if you have them)
            all_doc_vecs = torch.cat([pos_vecs.unsqueeze(1), neg_vecs.unsqueeze(1)], dim=1)  # (batch, 2, hidden_dim)

            # Compute cosine similarity between each query and its docs
            sims = torch.nn.functional.cosine_similarity(
                query_vecs.unsqueeze(1), all_doc_vecs, dim=2
            )  # (batch, 2)

            # For each query, is the positive doc in the top-k?
            topk = sims.topk(k, dim=1).indices  # (batch, k)
            # The positive doc is always at index 0 in all_doc_vecs
            correct += (topk == 0).any(dim=1).sum().item()
            total += batch_size

    return correct / total

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rnn_hidden_dim", type=int, default=256)
    parser.add_argument("--num_rnn_layers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--margin", type=float, default=0.28114883900596277)
    parser.add_argument("--dropout", type=float, default=0.17360380872013242)
    parser.add_argument("--lr", type=float, default=0.0018616575283672315)
    parser.add_argument("--freeze_embeddings", type=bool, default=True)
    parser.add_argument("--embedding_type", type=str, default="glove")
    parser.add_argument("--embedding_path", type=str, default="data/glove.6B.300d.txt")
    parser.add_argument("--emb_dim", type=int, default=300)
    parser.add_argument("--checkpoint_path", type=str, default=None)

    return parser.parse_args()

def main():
    
    # --- Hyperparameters ---
    args = parse_args()
    rnn_hidden_dim = args.rnn_hidden_dim
    num_rnn_layers = args.num_rnn_layers
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    margin = args.margin
    dropout = args.dropout
    lr = args.lr
    freeze_embeddings = args.freeze_embeddings
    embedding_type = args.embedding_type
    embedding_path = args.embedding_path
    emb_dim = args.emb_dim
    checkpoint_path = args.checkpoint_path
    rnn_type = "gru"
    vocab_path = "train_cbow/tkn_words_to_ids.pkl"

    # --- Embedding Configuration ---
    embedding_type = 'glove'  # 'cbow', 'glove', or 'random'
    embedding_path = 'data/glove.6B.300d.txt'  # Path to your GloVe file or CBOW checkpoint
    emb_dim = 300  # Changed to 300 for GloVe or 128 for CBOW
    freeze_embeddings = True  # Set to False to enable fine-tuning
    cbow_vocab_size = 30000  # Hardcoded CBOW vocabulary size (this does not matter for GloVe)
    if embedding_type == 'cbow':
        vocab_size = cbow_vocab_size
    else:
        vocab_size = None

    # --- wandb setup ---
    run = wandb.init(
        project="two-tower-msmarco",
        config={
            "emb_dim": emb_dim,
            "rnn_hidden_dim": rnn_hidden_dim,
            "num_rnn_layers": num_rnn_layers,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "margin": margin,
            "learning_rate": lr,
            "dropout": dropout,
            "embedding_type": embedding_type,
            "freeze_embeddings": freeze_embeddings,
            "rnn_type": rnn_type,
        }
    )

    # --- Hyperparameters from wandb sweep agent ---
    config = wandb.config 
    rnn_hidden_dim = config.rnn_hidden_dim
    num_rnn_layers = config.num_rnn_layers
    batch_size = config.batch_size
    margin = config.margin
    dropout = config.dropout
    lr = config.learning_rate
    freeze_embeddings = config.freeze_embeddings

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Data Preparation ---
    print("Loading dataset splits...")
    train_split, val_split, test_split = get_marco_ds_splits()
    print("Generating triplets...")
    train_triplets = generate_triplets(train_split, max_negatives_per_query=6)
    print(f"Number of training triplets: {len(train_triplets)}")
    train_dataset = MARCOTripletDataset(train_triplets, vocab_path=vocab_path, vocab_size=vocab_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    val_triplets = generate_triplets(val_split, max_negatives_per_query=2)
    val_dataset = MARCOTripletDataset(val_triplets, vocab_path=vocab_path, vocab_size=vocab_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    test_triplets = generate_triplets(test_split, max_negatives_per_query=2)
    test_dataset = MARCOTripletDataset(test_triplets, vocab_path=vocab_path, vocab_size=vocab_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # --- Load Vocabulary and Embedding Layer ---
    print("Loading vocabulary...")
    vocab = load_vocab(vocab_path=vocab_path, vocab_size=vocab_size)
    
    print(f"Loading {embedding_type} embeddings...")
    embedding_layer = load_embedding_layer(
        embedding_type=embedding_type,
        vocab=vocab,
        emb_dim=emb_dim,
        embedding_path=embedding_path,
        freeze=freeze_embeddings,
        cbow_vocab_size=cbow_vocab_size
    )
    print("Embedding requires_grad:", embedding_layer.weight.requires_grad)
    
    model = TwoTowerModel(embedding_layer, emb_dim, rnn_hidden_dim, num_rnn_layers, rnn_type=rnn_type, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # --- Checkpoint loading ---
    start_epoch = 0
    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        print(f"Resuming from epoch {start_epoch}")

    # --- Training Loop ---
    print("Starting training...")
    best_recall = 0
    patience = 3  # Number of epochs to wait for improvement
    epochs_no_improve = 0
    wandb_run_name = run.name or run.id
    save_path = os.path.join("checkpoints", wandb_run_name)
    os.makedirs(save_path, exist_ok=True)
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = train_one_epoch(model, train_loader, optimizer, device, margin=margin)
        print(f"Average loss: {epoch_loss:.4f}")

        val_loss = validate_one_epoch(model, val_loader, device, margin=margin)
        print(f"Validation loss: {val_loss:.4f}")
        wandb.log({"epoch": epoch+1, "train_loss": epoch_loss, "val_loss": val_loss})

        # Calculate and log Recall@1
        recall = recall_at_k(model, val_loader, device, k=1)
        print(f"Recall@1: {recall:.4f}")
        wandb.log({"epoch": epoch+1, "val_recall@1": recall})

        # Save checkpoint
        checkpoint = {
            "epoch": epoch+1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss,
            "model_args": {
                "embedding_dim": emb_dim,
                "hidden_dim": rnn_hidden_dim,
                "num_layers": num_rnn_layers,
                "dropout": dropout,
                "rnn_type": rnn_type,
                "vocab_size": vocab_size,
                # Add more if needed
            },
        }
        torch.save(checkpoint, f"{save_path}/two_tower_checkpoint_epoch_{epoch+1}.pth")
        print(f"Checkpoint saved for epoch {epoch+1}")

        # --- Early Stopping Logic ---
        if recall > best_recall:
            best_recall = recall
            epochs_no_improve = 0
            # Save the best model
            best_checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_args": checkpoint["model_args"],
            }
            torch.save(best_checkpoint, f'{save_path}/best_two_tower_model.pth')
            print("Best model saved.")
        else:
            epochs_no_improve += 1
            print(f"No improvement in recall for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    # After training:
    test_loss = validate_one_epoch(model, test_loader, device, margin=margin)
    print(f"Test loss: {test_loss:.4f}")
    wandb.log({"test_loss": test_loss})

    # Calculate and log Test Recall@1
    test_recall_1 = recall_at_k(model, test_loader, device, k=1)
    print(f"Test Recall@1: {test_recall_1:.4f}")
    wandb.log({"test_recall@1": test_recall_1})

    # Save final model
    print("Saving final model...")
    torch.save(model.state_dict(), 'two_tower_model_final.pth')
    print("Training complete!")
    wandb.finish()

if __name__ == "__main__":
    main() 