import argparse
import os.path as osp
import random
import sys
from pathlib import Path

import networkx as nx
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.data import Data, DataLoader
from torch_geometric.data.in_memory_dataset import InMemoryDataset
from torch_geometric.datasets import PascalPF
from torch_geometric.nn import GINEConv
from torch_geometric.utils import from_networkx

# Add parent directory to Python path
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

# from dgmc.models import DGMC, SplineCNN
from random_graph import RandomGraphDataset

from dgmc.models.dgmc import DGMC
from dgmc.models.gin import GIN

parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, default=256)
parser.add_argument("--rnd_dim", type=int, default=8)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_steps", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=200)
args = parser.parse_args()


class GraphNmrDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=True)
        self.load(self.processed_paths[0])
        self.pairs = torch.load(self.processed_paths[1])

    @property
    def raw_files_names(self):
        return ["1PWL_graph.gml", "1PWL_fold_graph.gml"]

    @property
    def processed_file_names(self):
        return ["data.pt", "pairs.pt"]

    def process(self):
        data_list = []
        raw_path = Path(self.raw_dir)

        graph_s: nx.Graph = nx.read_gml(raw_path / "1PLW_graph.gml")
        graph_t: nx.Graph = nx.read_gml(raw_path / "1PLW_fold_graph.gml")

        for _, _, d in graph_s.edges(data=True):
            d["weight"] = float(d["weight"])

        for _, _, d in graph_t.edges(data=True):
            d["weight"] = float(d["weight"])

        data_s = from_networkx(graph_s, group_edge_attrs=["weight"])
        data_t = from_networkx(graph_t, group_edge_attrs=["weight"])

        pairs = [(0, 1)]

        data_list.append(data_s)
        data_list.append(data_t)

        self.save(data_list, self.processed_paths[0])
        torch.save(pairs, self.processed_paths[1])


transform = T.Constant()
train_dataset = RandomGraphDataset(num_nodes=50, p=0.3, is_connected=True, transform=transform)
train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, follow_batch=["x_s", "x_t"])

# Graph NMR
test_dataset = GraphNmrDataset(root="./datasets/1PLW", transform=T.Constant())


device = "cuda" if torch.cuda.is_available() else "cpu"

psi_1 = GIN(in_channels=1, out_channels=8, num_layers=3)
psi_2 = GIN(in_channels=args.rnd_dim, out_channels=args.rnd_dim, num_layers=3)

model = DGMC(psi_1, psi_2, num_steps=args.num_steps).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train():
    model.train()

    total_loss = total_examples = total_correct = 0
    for i, data in enumerate(train_loader):
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
        total_examples += y.size(1)

    return total_loss / len(train_loader), total_correct / total_examples


@torch.no_grad()
def test(dataset):
    model.eval()

    correct = num_examples = 0
    for pair in dataset.pairs:
        data_s, data_t = dataset[pair[0]], dataset[pair[1]]
        data_s, data_t = data_s.to(device), data_t.to(device)
        S_0, S_L = model(
            data_s.x,
            data_s.edge_index,
            data_s.edge_attr,
            None,
            data_t.x,
            data_t.edge_index,
            data_t.edge_attr,
            None,
        )
        y = torch.arange(data_s.num_nodes, device=device)
        y = torch.stack([y, y], dim=0)
        correct += model.acc(S_L, y, reduction="sum")
        num_examples += y.size(1)

    return correct / num_examples


for epoch in range(1, args.epochs):
    loss, acc = train()
    print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.2f}")

    accs = test(test_dataset)
    print(100 * accs)
