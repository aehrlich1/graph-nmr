import random
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, DataLoader
from torch_geometric.data.in_memory_dataset import InMemoryDataset
from torch_geometric.utils import from_networkx
from torch_geometric.utils.convert import to_networkx


def assign_numbers(list1, list2):
    # Check for duplicates within each list
    assert len(list1) == len(set(list1)), "list1 has duplicates"
    assert len(list2) == len(set(list2)), "list2 has duplicates"

    # Create unified mapping
    all_strings = set(list1) | set(list2)
    mapping = {s: i for i, s in enumerate(sorted(all_strings))}

    # Apply mapping to both lists
    numbered1 = [mapping[s] for s in list1]
    numbered2 = [mapping[s] for s in list2]

    return numbered1, numbered2


class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_s":
            return self.x_s.size(0)
        if key == "edge_index_t":
            return self.x_t.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class GraphNmrDataset1k(InMemoryDataset):
    def __init__(
        self, root, transform=None, pre_transform=None, pre_filter=None, force_reload=True
    ):
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        data_list = []
        raw_path = Path(self.raw_dir)

        for subdir in raw_path.iterdir():
            if subdir.is_dir() == False:
                continue
            #assert subdir.is_dir(), "raw_path contains non directory files"
            for file in subdir.iterdir():
                if file.name.endswith("graph_fold.gml"):
                    graph_t: nx.Graph = nx.read_gml(file, label="id")
                elif file.name.endswith("graph.gml"):
                    graph_s: nx.Graph = nx.read_gml(file, label="id")


            assert not nx.is_empty(graph_s), "source graph is empty"
            assert not nx.is_empty(graph_t), "target graph is empty"

            for _, _, d in graph_s.edges(data=True):
                d["weight"] = float(d["weight"])

            for _, _, d in graph_t.edges(data=True):
                d["weight"] = float(d["weight"])

            for _, d in graph_s.nodes(data=True):
                d["chem_shift"] = float(d["chem_shift"])
                d["label"] = str(d["label"])

            for _, d in graph_t.nodes(data=True):
                d["plDDT"] = float(d["plDDT"])
                d["label"] = str(d["label"])

            data_s = from_networkx(graph_s, group_edge_attrs=["weight"])
            data_t = from_networkx(graph_t, group_edge_attrs=["weight"])

            y_s, y_t = assign_numbers(data_s.label, data_t.label)

            data: PairData = PairData(
                x_s=data_s.chem_shift,
                edge_index_s=data_s.edge_index,
                edge_attr_s=data_s.edge_attr,
                label_s=data_s.label,
                y_s=torch.tensor(y_s, dtype=torch.long),
                x_t=data_t.plDDT,
                edge_index_t=data_t.edge_index,
                edge_attr_t=data_t.edge_attr,
                label_t=data_t.label,
                y_t=torch.tensor(y_t, dtype=torch.long),
                protein=subdir.name,
            )

            # Form pairs based on which source graph to match to target graph
            data_list.append(data)

        self.save(data_list, self.processed_paths[0])


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

        # Form pairs based on which source graph to match to target graph
        pairs = [(0, 1)]

        data_list.append(data_s)
        data_list.append(data_t)

        self.save(data_list, self.processed_paths[0])
        torch.save(pairs, self.processed_paths[1])


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
