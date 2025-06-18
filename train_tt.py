import torch
import torch.nn as nn
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from two_tower_utils import load_frozen_embedding_from_cbow
from two_tower_model import TwoTowerModel
from data_utils import get_marco_ds_splits, generate_triplets, MARCOTripletDataset
import wandb
import os
from tqdm import tqdm

torch.manual_seed(42)
torch.cuda.manual_seed(42)

def collate_fn(batch):
    """
    Collate function for batching triplets with padding.
    batch: list of (query_tokens, pos_tokens, neg_tokens)
    Return: query_tensor, pos_tensor, neg_tensor (all shape: [batch, seq_len])
    """
    # Separate out queries, positives, negatives
    queries, positives, negatives = zip(*batch)
    
    # Convert each to a list of tensors
    queries = [torch.tensor(q, dtype=torch.long) for q in queries]
    positives = [torch.tensor(p, dtype=torch.long) for p in positives]
    negatives = [torch.tensor(n, dtype=torch.long) for n in negatives]
    
    # Pad each group
    queries_padded = pad_sequence(queries, batch_first=True, padding_value=0)
    positives_padded = pad_sequence(positives, batch_first=True, padding_value=0)
    negatives_padded = pad_sequence(negatives, batch_first=True, padding_value=0)
    
    return queries_padded, positives_padded, negatives_padded

def train_one_epoch(model, dataloader, optimizer, device, margin=0.2):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc='Training', leave=False):
        # Unpack batch and move to device
        query_tokens, pos_tokens, neg_tokens = batch
        query_tokens = query_tokens.to(device)
        pos_tokens = pos_tokens.to(device)
        neg_tokens = neg_tokens.to(device)
        
        # Forward pass through model
        query_vec, pos_doc_vec, neg_doc_vec = model(query_tokens, pos_tokens, neg_tokens)
        
        # Compute cosine similarities
        pos_sim = torch.nn.functional.cosine_similarity(query_vec, pos_doc_vec, dim=1)
        neg_sim = torch.nn.functional.cosine_similarity(query_vec, neg_doc_vec, dim=1)
        
        # Compute triplet loss: encourage pos_sim > neg_sim by a margin
        # Use relu to implement loss = max(neg_sim - pos_sim + margin, 0)
        triplet_loss = torch.nn.functional.relu(neg_sim - pos_sim + margin).mean()
        
        # Backpropagation and optimizer step
        optimizer.zero_grad()
        triplet_loss.backward()
        optimizer.step()
        
        # Accumulate loss
        total_loss += triplet_loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def validate_one_epoch(model, dataloader, device, margin=0.2):
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating', leave=False):
            query_tokens, pos_tokens, neg_tokens = batch
            query_tokens = query_tokens.to(device)
            pos_tokens = pos_tokens.to(device)
            neg_tokens = neg_tokens.to(device)
            query_vec, pos_doc_vec, neg_doc_vec = model(query_tokens, pos_tokens, neg_tokens)
            pos_sim = torch.nn.functional.cosine_similarity(query_vec, pos_doc_vec, dim=1)
            neg_sim = torch.nn.functional.cosine_similarity(query_vec, neg_doc_vec, dim=1)
            triplet_loss = torch.nn.functional.relu(neg_sim - pos_sim + margin).mean()
            total_loss += triplet_loss.item()
            num_batches += 1
    return total_loss / num_batches

def recall_at_k(model, dataloader, device, k=5):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Recall@{k}', leave=False):
            query_tokens, pos_tokens, neg_tokens = batch
            query_tokens = query_tokens.to(device)
            pos_tokens = pos_tokens.to(device)
            neg_tokens = neg_tokens.to(device)
            batch_size = query_tokens.size(0)

            # Encode queries and all docs (positive and negative)
            query_vecs = model.encode_query(query_tokens)  # (batch, hidden_dim)
            pos_vecs = model.encode_document(pos_tokens)   # (batch, hidden_dim)
            neg_vecs = model.encode_document(neg_tokens)   # (batch, hidden_dim)

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

def main():
    # --- Hyperparameters ---
    vocab_size = 30000
    emb_dim = 128
    rnn_hidden_dim = 128
    num_rnn_layers = 1
    batch_size = 32
    num_epochs = 5
    margin = 0.2
    cbow_checkpoint = "train_cbow/checkpoints/cbow_2025_06_17__13_43_21.epoch_10.pth"
    vocab_path = "train_cbow/tkn_words_to_ids.pkl"
    lr = 1e-3
    checkpoint_path = None  # Set to a checkpoint file to resume, or None to start fresh

    # --- wandb setup ---
    wandb.init(
        project="two-tower-msmarco",
        config={
            "vocab_size": vocab_size,
            "emb_dim": emb_dim,
            "rnn_hidden_dim": rnn_hidden_dim,
            "num_rnn_layers": num_rnn_layers,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "margin": margin,
            "learning_rate": lr,
        }
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Data Preparation ---
    print("Loading dataset splits...")
    train_split, val_split, test_split = get_marco_ds_splits()
    print("Generating triplets...")
    train_triplets = generate_triplets(train_split, max_negatives_per_query=1)
    print(f"Number of training triplets: {len(train_triplets)}")
    train_dataset = MARCOTripletDataset(train_triplets, vocab_path=vocab_path, vocab_size=vocab_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    val_triplets = generate_triplets(val_split, max_negatives_per_query=1)
    val_dataset = MARCOTripletDataset(val_triplets, vocab_path=vocab_path, vocab_size=vocab_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    test_triplets = generate_triplets(test_split, max_negatives_per_query=1)
    test_dataset = MARCOTripletDataset(test_triplets, vocab_path=vocab_path, vocab_size=vocab_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # --- Model and Optimizer ---
    print("Loading pretrained embeddings...")
    embedding_layer = load_frozen_embedding_from_cbow(cbow_checkpoint, vocab_size, emb_dim)
    model = TwoTowerModel(embedding_layer, emb_dim, rnn_hidden_dim, num_rnn_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = train_one_epoch(model, train_loader, optimizer, device, margin=margin)
        print(f"Average loss: {epoch_loss:.4f}")

        val_loss = validate_one_epoch(model, val_loader, device, margin=margin)
        print(f"Validation loss: {val_loss:.4f}")
        wandb.log({"epoch": epoch+1, "train_loss": epoch_loss, "val_loss": val_loss})

        # Save checkpoint
        checkpoint = {
            "epoch": epoch+1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss,
        }
        torch.save(checkpoint, f"two_tower_checkpoint_epoch_{epoch+1}.pth")
        print(f"Checkpoint saved for epoch {epoch+1}")

        # Calculate and log Recall@5
        recall = recall_at_k(model, val_loader, device, k=5)
        print(f"Recall@5: {recall:.4f}")
        wandb.log({"epoch": epoch+1, "val_recall@5": recall})

    # After training:
    test_loss = validate_one_epoch(model, test_loader, device, margin=margin)
    print(f"Test loss: {test_loss:.4f}")
    wandb.log({"test_loss": test_loss})

    # Calculate and log Test Recall@5
    test_recall_5, test_recall_10 = recall_at_k(model, test_loader, device, k=5), recall_at_k(model, test_loader, device, k=10)
    print(f"Test Recall@5: {test_recall_5:.4f}, Test Recall@10: {test_recall_10:.4f}")
    wandb.log({"test_recall@5": test_recall_5, "test_recall@10": test_recall_10})

    # Save final model
    print("Saving final model...")
    torch.save(model.state_dict(), 'two_tower_model_final.pth')
    print("Training complete!")
    wandb.finish()

if __name__ == "__main__":
    main() 