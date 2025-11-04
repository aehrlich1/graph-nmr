import os.path as osp
import sys
from pathlib import Path

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.data.in_memory_dataset import InMemoryDataset
from torch_geometric.utils import from_networkx, is_undirected
from tqdm import tqdm

# Add parent directory to Python path
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))


class GraphNmrDataset(InMemoryDataset):
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

        for subdir in tqdm(raw_path.iterdir()):
            if not subdir.is_dir():
                print(f"Skipped {subdir.name} because it is not a subdirectory.")
                continue
            for file in subdir.iterdir():
                if file.name.endswith("fold_graph.gml"):
                    graph_t: nx.Graph = nx.read_gml(file, label="id")
                elif file.name.endswith("graph.gml"):
                    graph_s: nx.Graph = nx.read_gml(file, label="id")

            # TODO: assert that graph_s and graph_t are not empty

            try:
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
            except Exception as err:
                print(f"Error at protein ID: {subdir.name}")
                print(f"Failed to read graph attributes: {err}")

            data_s = from_networkx(graph_s, group_edge_attrs=["weight"])
            data_t = from_networkx(graph_t, group_edge_attrs=["weight"])

            assert data_s.is_undirected()
            assert data_t.is_undirected()

            missing = [item for item in data_s.label if item not in data_t.label]

            if missing:
                print(
                    f"Warning: Source labels {missing} not found in target labels, for protein {subdir.name}"
                )

            y_s = torch.arange(data_s.size(0), dtype=torch.long)
            y_t = generate_y_t(data_s.label, data_t.label)

            # Extract Amino Acid 3 letter code from label
            amino_acid_s = extract_amino_acid(data_s.label)
            amino_acid_t = extract_amino_acid(data_t.label)

            if self.transform is not None:
                data_s = self.transform(data_s)
                data_t = self.transform(data_t)

            data: PairData = PairData(
                x_s=data_s.chem_shift.view(-1, 1),
                amino_acid_s=amino_acid_to_onehot(amino_acid_s),
                edge_index_s=data_s.edge_index,
                edge_attr_s=data_s.edge_attr,
                label_s=data_s.label,
                y_index_s=y_s,
                x_t=data_t.plDDT.view(-1, 1),
                amino_acid_t=amino_acid_to_onehot(amino_acid_t),
                edge_index_t=data_t.edge_index,
                edge_attr_t=data_t.edge_attr,
                label_t=data_t.label,
                y_t=y_t,
                num_nodes=y_s.size(0),
                protein=subdir.name,
            )

            # Apply pre_transform if provided
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # Form pairs based on which source graph to match to target graph
            data_list.append(data)

        self.save(data_list, self.processed_paths[0])


class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_s":
            return self.x_s.size(0)
        if key == "edge_index_t":
            return self.x_t.size(0)
        return super().__inc__(key, value, *args, **kwargs)


def extract_amino_acid(labels: list[str]) -> list[str]:
    amino_acids = []
    for label in labels:
        words = label.split()
        if len(words) != 3:
            raise ValueError(f"Label must contain exactly 3 words, got {len(words)}: '{label}'")
        amino_acids.append(words[1])

    return amino_acids


def amino_acid_to_onehot(amino_acids: list[str]) -> torch.Tensor:
    # Standard 20 amino acids in alphabetical order
    # asdf
    AA_VOCAB = [
        "ALA",
        "CYS",
        "ASP",
        "GLU",
        "PHE",
        "GLY",
        "HIS",
        "ILE",
        "LYS",
        "LEU",
        "MET",
        "ASN",
        "PRO",
        "GLN",
        "ARG",
        "SER",
        "THR",
        "VAL",
        "TRP",
        "TYR",
    ]

    aa_to_idx = {aa: idx for idx, aa in enumerate(AA_VOCAB)}

    # Convert amino acids to indices
    indices = []
    for aa in amino_acids:
        if aa not in aa_to_idx:
            raise ValueError(f"Unknown amino acid code: {aa}. Expected one of {AA_VOCAB}")
        indices.append(aa_to_idx[aa])

    # Create one-hot encoding
    num_amino_acids = len(amino_acids)
    onehot = torch.zeros(num_amino_acids, len(AA_VOCAB), dtype=torch.float)
    onehot[torch.arange(num_amino_acids), indices] = 1.0

    # return torch.tensor(indices, dtype=torch.float)
    return onehot


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

    # TODO assert numbered2 is a subset of numbered1

    return torch.tensor(numbered1, dtype=torch.long), torch.tensor(numbered2, dtype=torch.long)


def generate_y_t(label_s, label_t):
    y_s, y_t = assign_numbers(label_s, label_t)

    # perm = torch.argsort(y_t)[torch.argsort(torch.argsort(y_s))]
    # y_s_hat = torch.arange(y_s.size(0))
    # y_t_hat = y_s_hat[perm]

    # return y_t_hat.long()
    try:
        perm = torch.tensor([(y_t == v).nonzero(as_tuple=True)[0].item() for v in y_s])
    except Exception:
        print("Returning y_t with zeros.")
        return torch.zeros_like(y_s)

    return perm.long()


def pre_transform_to_ones(data):
    """
    Pre-transform function that converts both x_s and x_t values to ones.
    This normalizes node features to uniform values while preserving graph structure.
    """
    if hasattr(data, "x_s") and data.x_s is not None:
        data.x_s = torch.ones_like(data.x_s)

    if hasattr(data, "x_t") and data.x_t is not None:
        data.x_t = torch.ones_like(data.x_t)

    return data
