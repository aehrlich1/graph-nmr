import random

import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_networkx
from torch_geometric.utils.convert import to_networkx


class RandomGraphDataset(torch.utils.data.Dataset):
    def __init__(self, num_nodes=25, p=0.4, is_connected=True, transform=None):
        self.num_nodes = num_nodes
        self.p = p
        self.is_connected = is_connected
        self.transform = transform

    def __len__(self):
        return 128  # 1024

    def __getitem__(self, idx):
        G_s = generate_fully_connected_graph(num_nodes=self.num_nodes, p=self.p)
        G_t = remove_edges_from_graph(G_s, p=0.1)
        G_t = perturb_edge_weights(G_t, noise=0.1)

        # print(f"G_s.edges(data=True) = {G_s.edges(data=True)}")
        # print(f"G_t.edges(data=True) = {G_t.edges(data=True)}")

        data_s = from_networkx(G_s, group_edge_attrs="all")
        data_t = from_networkx(G_t, group_edge_attrs="all")

        # draw_data(data_s)
        # draw_data(data_t)
        data_s["y_index"] = torch.arange(data_s.num_nodes)
        data_t["y"] = torch.arange(data_t.num_nodes)

        if self.transform is not None:
            data_s = self.transform(data_s)
            data_t = self.transform(data_t)

        data = Data(num_nodes=data_s.num_nodes)
        for key in data_s.keys():
            data[f"{key}_s"] = data_s[key]
        for key in data_t.keys():
            data[f"{key}_t"] = data_t[key]

        return data


def remove_edges_from_graph(G, p) -> nx.Graph:
    """
    Removes each edge from G with probability p, ensuring the graph remains connected.
    Returns a modified copy of G.
    """
    G = G.copy()
    edges = list(G.edges())
    random.shuffle(edges)
    for u, v in edges:
        if random.random() < p:
            G.remove_edge(u, v)
            if not nx.is_connected(G):
                G.add_edge(u, v)
    return G


def perturb_edge_weights(G, noise=0.05) -> nx.Graph:
    """
    Add uniform random noise betwen [-noise, noise] to the edge weights.
    """
    G = G.copy()
    for u, v, data in G.edges(data=True):
        noise = random.uniform(-noise, noise)
        data["weight"] += noise

    return G


def generate_fully_connected_graph(num_nodes, p=0.4) -> nx.Graph:
    G = nx.erdos_renyi_graph(num_nodes, p, directed=False)
    for _ in range(100000):
        if nx.is_connected(G):
            for u, v in G.edges():
                G[u][v]["weight"] = random.uniform(0, 10)
            return G
    raise RuntimeError("Could not create fully connected graph.")


def draw_graph(G):
    plt.figure(figsize=(4, 4))
    nx.draw(G, with_labels=True, node_color="lightblue", node_size=50, font_weight="bold")
    plt.show()


def draw_data(data):
    G = to_networkx(data, to_undirected=True)
    draw_graph(G)


train_dataset = RandomGraphDataset(20, 0.5)
train_loader = DataLoader(train_dataset, 1, shuffle=True, follow_batch=["x_s", "x_t"])

print(train_loader)
