# poetry run torchrun --nproc_per_node=4 --nnodes=1 --master_addr="localhost" --master_port=12345 src/test_distributed.py --epochs 50

import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


# Custom Dataset (Example using random data)
class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        # For simplicity, targets are also random
        return self.data[index], self.data[index]

    def __len__(self):
        return self.len


# Simple model
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


# Training function
def train(args):
    # Get rank and world size
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Initialize the process group
    dist.init_process_group(backend="gloo")

    # Configure logger to include rank information
    logger.remove()  # Remove default logger
    logger.add(
        sys.stderr,
        format=f"<green>{{time}}</green> | <level>{{level}}</level> | Rank {rank} | <cyan>{{message}}</cyan>",
        level="INFO",
    )

    logger.info(f"Starting training on rank {rank}. World size: {world_size}")

    # Set the seed for reproducibility
    torch.manual_seed(0)

    # Create the model
    model = ToyModel()

    # Wrap the model with DDP
    ddp_model = DDP(model)

    # Create dataset and distributed sampler
    dataset = RandomDataset(size=10, length=64)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)  # Shuffle data for each epoch
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Log batch-level information
            logger.info(
                f"Epoch [{epoch+1}/{args.epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}"
            )

        # Average loss across all batches
        avg_loss = epoch_loss / len(dataloader)

        # Log epoch-level information
        logger.info(
            f"Epoch [{epoch+1}/{args.epochs}] completed. Average Loss: {avg_loss:.4f}"
        )

    logger.info(f"Training completed on rank {rank}.")

    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="PyTorch DDP Example with DataLoader and CLI Arguments"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="number of total epochs to run"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="batch size per process"
    )
    args = parser.parse_args()

    train(args)
