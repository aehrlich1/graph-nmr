import argparse
import os.path as osp
import sys

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, DataLoader

# Add parent directory to Python path
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

# from dgmc.models import DGMC, SplineCNN
from dgmc.models.dgmc import DGMC
from dgmc.models.gine import GINE
from examples.random_graph import GraphNmrDataset, GraphNmrDataset1k, RandomGraphDataset

parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, default=256)
parser.add_argument("--rnd_dim", type=int, default=8)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_steps", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=200)
args = parser.parse_args()


dataset = GraphNmrDataset1k(root="./datasets/nmr_graphs_1k", force_reload=False)

# Randomly split dataset into training and test
dataset = dataset.shuffle()
train_dataset = dataset[:700]
test_dataset = dataset[700:]

train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, follow_batch=["x_s", "x_t"])
test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, follow_batch=["x_s", "x_t"])

device = "cuda" if torch.cuda.is_available() else "cpu"

psi_1 = GINE(in_channels=1, out_channels=8, num_layers=3)
psi_2 = GINE(in_channels=args.rnd_dim, out_channels=args.rnd_dim, num_layers=3)

model = DGMC(psi_1, psi_2, num_steps=args.num_steps).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train():
    model.train()

    total_loss = total_examples = total_correct = 0
    for i, data in enumerate(train_loader):
        # Override x_s, x_t with ones
        data.x_s = torch.ones(data.x_s.size(0), 1)
        data.x_t = torch.ones(data.x_t.size(0), 1)

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

        # TODO fix ground truth matchings
        y = torch.stack([data.y_s, data.y_t[: len(data.y_s)]], dim=0)

        print(f"data.y_s.max(): {data.y_s.max()}")
        print(f"data.x_s.size(0): {data.x_s.size(0)}")
        print(f"data.y_t.max(): {data.y_t.max()}")
        print(f"data.x_t.size(0): {data.x_t.size(0)}")

        loss = model.loss(S_0, y)
        loss = model.loss(S_L, y) + loss if model.num_steps > 0 else loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += model.acc(S_L, y, reduction="sum")
        total_examples += y.size(1)

    return total_loss / len(train_loader), total_correct / total_examples


@torch.no_grad()
def test(dataset):
    model.eval()

    correct = num_examples = 0
    for data in dataset:
        # Override x_s, x_t with ones
        data.x_s = torch.ones(data.x_s.size(0), 1)
        data.x_t = torch.ones(data.x_t.size(0), 1)

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
        y = torch.stack([data.y_s, data.y_t], dim=0)
        correct += model.acc(S_L, y, reduction="sum")
        num_examples += y.size(1)

    return correct / num_examples


for epoch in range(1, args.epochs):
    loss, acc = train()
    print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.2f}")

    accs = test(test_dataset)
    print(100 * accs)
