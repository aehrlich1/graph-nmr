import argparse
import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from dgmc.datasets import GraphNmrDataset
from dgmc.models import DGMC, GIN, GINE

parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, default=256)
parser.add_argument("--rnd_dim", type=int, default=8)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_steps", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.002)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=10)
args = parser.parse_args()

script_dir = Path(__file__).parent
dataset_root = script_dir.parent / "data" / "nmr_graphs_4k_filtered_mismatches"
dataset = GraphNmrDataset(root=dataset_root, force_reload=True)

dataset = dataset.shuffle()

# Split dataset into train and test
num_graphs = len(dataset)
split_index = max(1, min(int(0.8 * num_graphs), num_graphs - 1))
train_dataset = dataset[:split_index]
test_dataset = dataset[split_index:]

train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, follow_batch=["x_s", "x_t"])
test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, follow_batch=["x_s", "x_t"])

device = "cuda" if torch.cuda.is_available() else "cpu"

psi_1 = GIN(in_channels=20, out_channels=32, num_layers=2)
psi_2 = GIN(in_channels=args.rnd_dim, out_channels=args.rnd_dim, num_layers=2)

model = DGMC(psi_1, psi_2, num_steps=args.num_steps).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train():
    model.train()

    total_loss = total_examples = total_correct = total_correct_k = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        S_0, S_L = model(
            data.x_s,
            data.edge_index_s,
            data.edge_attr_s,
            data.x_s_batch,
            data.x_t,
            data.edge_index_t,
            data.edge_attr_t,
            data.x_t_batch,
        )

        y = torch.stack([data.y_index_s, data.y_t], dim=0)
        loss = model.loss(S_0, y)
        loss = model.loss(S_L, y) + loss if model.num_steps > 0 else loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += model.acc(S_L, y, reduction="sum")
        total_correct_k += model.hits_at_k(3, S_L, y, reduction="sum")
        total_examples += y.size(1)

    return (
        total_loss / len(train_loader),
        total_correct / total_examples,
        total_correct_k / total_examples,
    )


@torch.no_grad()
def test(dataset):
    model.eval()

    correct = num_examples = 0
    for data in dataset:
        data = data.to(device)
        S_0, S_L = model(
            data.x_s,
            data.edge_index_s,
            data.edge_attr_s,
            None,
            data.x_t,
            data.edge_index_t,
            data.edge_attr_t,
            None,
        )

        y = torch.stack([data.y_index_s, data.y_t], dim=0)
        correct += model.acc(S_L, y, reduction="sum")
        num_examples += y.size(1)

    return correct / num_examples


for epoch in range(1, args.epochs):
    loss, acc, acc_k = train()
    print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.2f}, Acc_k: {acc_k:.2f}")

    accs = test(test_dataset)
    print(100 * accs)
