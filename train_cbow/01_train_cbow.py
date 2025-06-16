# train_cbow.py

import torch
import tqdm
import wandb
import datetime
from dataset import CBOW as CBOWDataset
from cbow_model import CBOW as CBOWModel
from evaluate import topk


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_dataloader(batch_size: int = 256, vocab_size=None) -> tuple[CBOWDataset, torch.utils.data.DataLoader]:
    dataset = CBOWDataset(vocab_size=vocab_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, dataloader


def train(model, dataloader, optimizer, criterion, device, run_name, num_epochs=5):
    model.to(device)
    wandb.init(project='mlx-cbow', name=run_name)

    for epoch in range(1, num_epochs + 1):
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

        save_checkpoint(model, run_name, epoch)

    wandb.finish()


def save_checkpoint(model, run_name: str, epoch: int):
    filename = f'checkpoints/{run_name}.epoch_{epoch}.pth'
    torch.save(model.state_dict(), filename)

    artifact = wandb.Artifact('model-weights', type='model')
    artifact.add_file(filename)
    wandb.log_artifact(artifact)


def main():
    torch.manual_seed(42)
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    run_name = f'cbow_{timestamp}'
    device = get_device()

    vocab_size = 30000
    batch_size = 256
    dataset, dataloader = get_dataloader(batch_size=batch_size, vocab_size=vocab_size)

    print(f"Vocab Size: {vocab_size} out of a possible {len(dataset.int_to_vocab)}")

    model = CBOWModel(vocab_size=vocab_size, emb_dim=128)
    print('Model parameters:', sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = torch.nn.CrossEntropyLoss()

    train(model, dataloader, optimizer, criterion, device, run_name)


if __name__ == '__main__':
    main()
