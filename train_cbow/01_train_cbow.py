# train_cbow.py
import argparse
import datetime
import os

import torch
import tqdm
import wandb
from cbow_model import CBOW as CBOWModel
from dataset import CBOW as CBOWDataset
from evaluate import topk


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_dataloader(batch_size: int = 256, vocab_size=None, context_size=3) -> tuple[CBOWDataset, torch.utils.data.DataLoader]:
    dataset = CBOWDataset(vocab_size=vocab_size, context_size=context_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, dataloader

def save_checkpoint(model, optimizer, run_name: str, epoch: int):
    os.makedirs('checkpoints', exist_ok=True)
    filename = f'checkpoints/{run_name}.epoch_{epoch}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }, filename)

    artifact = wandb.Artifact('model-weights', type='model')
    artifact.add_file(filename)
    wandb.log_artifact(artifact)

def load_checkpoint(model, optimizer, run_name: str):
    checkpoints = [f for f in os.listdir('checkpoints') if f.startswith(run_name) and f.endswith('.pth')]
    if not checkpoints:
        return 0  # No checkpoint found, start from epoch 0
    # Find the latest checkpoint by epoch number
    latest = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0].replace('epoch', '')))
    path = os.path.join('checkpoints', latest)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Resumed from checkpoint: {path}")
    return checkpoint['epoch']

def train(model, dataloader, optimizer, criterion, device, run_name, num_epochs=5, start_epoch=1):
    model.to(device)
    wandb.init(project='mlx-cbow', name=run_name, resume="allow")

    for epoch in range(start_epoch, num_epochs + 1):
        progress = tqdm.tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        for step, (inputs, targets) in enumerate(progress):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.squeeze())
            loss.backward()
            optimizer.step()

            wandb.log({'loss': loss.item()})

            if step % 10_000 == 0:
                topk(model)

        save_checkpoint(model, optimizer, run_name, epoch)

    wandb.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default=None, help='Name for this training run (used for checkpoints and wandb)')
    args = parser.parse_args()

    torch.manual_seed(42)
    if args.run_name:
        run_name = args.run_name
    else:
        timestamp = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
        run_name = f'cbow_{timestamp}'
    device = get_device()

    vocab_size = 30000
    context_size = 3
    batch_size = 1024*8
    num_epochs = 10
    dataset, dataloader = get_dataloader(batch_size=batch_size, vocab_size=vocab_size, context_size=context_size)

    model = CBOWModel(vocab_size=vocab_size, emb_dim=128)
    print('Model parameters:', sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = torch.nn.CrossEntropyLoss()

    # Resume logic
    os.makedirs('checkpoints', exist_ok=True)
    start_epoch = 1
    # Try to find a checkpoint for this run_name
    checkpoints = [f for f in os.listdir('checkpoints') if f.startswith(run_name) and f.endswith('.pth')]
    if checkpoints:
        # Resume from latest checkpoint
        start_epoch = load_checkpoint(model, optimizer, run_name) + 1

    train(model, dataloader, optimizer, criterion, device, run_name, num_epochs)

if __name__ == '__main__':
    main()
