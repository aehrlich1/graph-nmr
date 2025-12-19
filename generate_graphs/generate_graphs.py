import random
import networkx as nx
import numpy as np
from pathlib import Path

def generate_new_node_name(G, base="AUGMENTED"):
    i = 0
    while f"{base}_{i}" in G:
        i += 1
    return f"{base}_{i}"

def add_random_edges(
    G,
    num_edges=5,
    edge_feature_fn=None,
    seed=None,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    G_new = G.copy()
    nodes = list(G_new.nodes)

    attempts = 0
    added = 0
    while added < num_edges and attempts < 10 * num_edges:
        u, v = random.sample(nodes, 2)
        attempts += 1

        if G_new.has_edge(u, v):
            continue


        distance= np.random.uniform(1.5,5.5)
        G_new.add_edge(u, v, weight=distance, label=f"{distance:.2f}")
        added += 1

    print(f"Added {added} edges")
    return G_new

def perturb_edge_distances(G, sigma=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)

    G_new = G.copy()
    for u, v, data in G_new.edges(data=True):
        if "weight" in data:
            data["weight"] += np.random.uniform(0, sigma)#np.random.normal(0, sigma)
            data["weight"] = max(data["weight"], 0.1)
    return G_new

def add_nodes_and_edges(
    G,
    num_new_nodes=1,
    max_edges_per_new_node=3,
    edge_feature_fn=None,
    connect_to_existing=True,
    seed=None,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    G_new = G.copy()

    # Freeze the original node set
    original_nodes = list(G.nodes)

    for i in range(num_new_nodes):
        new_node = f"HA THR {i}"

        G_new.add_node(
            new_node,
            chem_shift="1.66",
            chem_shift_error="0.03",
        )

        num_edges_target = random.randint(1, max_edges_per_new_node)
        to_connect = random.sample(original_nodes, num_edges_target)
        for target in to_connect:
            weight = np.random.uniform(1.5, 5.5)
            G_new.add_edge(new_node, target, weight=weight, label=f"{weight:.2f}")



    return G_new



def augment_graph_and_store(
    original_graph_path,
    output_root,
    num_new_nodes=1,
    max_edges_per_new_node=3,
    edge_feature_fn=None,
    connect_to_existing=True,
    seed=None, sigma=0.1,
):
    """
    Augment a NetworkX graph and store original + augmented versions.

    Folder structure:
    output_root/
        nodes_{num_new_nodes}_edges_{max_edges_per_new_node}/
            raw/
                <original_name>.gml
                <original_name>_fold_graph.gml

    Parameters
    ----------
    G : nx.Graph
        Original graph.
    original_graph_path : Path or str
        Path to the original graph file (used for naming).
    output_root : Path or str
        Root directory where augmented data will be stored.
    num_new_nodes : int
        Number of nodes to add.
    max_edges_per_new_node : int
        Maximum edges per new node.
    edge_feature_fn : callable or None
        Function returning edge feature (e.g. distance).
    connect_to_existing : bool
        Ensure new nodes connect to old graph.
    seed : int or None
        Random seed.
    """
    original_graph_path = Path(original_graph_path)
    output_root = Path(output_root)
    G = nx.read_gml(original_graph_path)

    # ---- create parameter-encoded folder name ----
    folder_name = f"nodes_{num_new_nodes}_edges_{max_edges_per_new_node}"
    raw_dir = output_root / folder_name / "raw"/"1HS7"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # ---- file naming ----
    original_name = original_graph_path.stem          # e.g. 1HS7_graph
    suffix = original_graph_path.suffix               # .gml

    augmented_name = original_name.replace(
        "_graph", "_fold_graph"
    ) + suffix

    original_out = raw_dir / (original_name + suffix)
    augmented_out = raw_dir / augmented_name

    # ---- save original graph ----
    nx.write_gml(G, original_out)

    # ---- augment graph ----
    #G_aug = add_random_edges(G, max_edges_per_new_node, seed=2)
    G_aug = perturb_edge_distances(G, sigma=sigma, seed=seed)
#    G_aug = add_nodes_and_edges(
 #       G,
 #       num_new_nodes=num_new_nodes,
  #      max_edges_per_new_node=max_edges_per_new_node,
   #     edge_feature_fn=edge_feature_fn,
    #    connect_to_existing=connect_to_existing,
     #   seed=seed,
   # )

    # ---- save augmented graph ----
    nx.write_gml(G_aug, augmented_out)

    return {
        "folder": raw_dir.parent,
        "original_graph": original_out,
        "augmented_graph": augmented_out,
    }

original_graph =  "/Users/florianwolf/PycharmProjects/graph-nmr2/data/nmr_graphs_2_easy/raw/1HS7/1HS7_graph.gml"
output_root = "/Users/florianwolf/PycharmProjects/graph-nmr2/data/uniform_dist"

augment_graph_and_store(original_graph_path=original_graph, output_root=output_root, num_new_nodes=0, max_edges_per_new_node=80, seed=2, sigma=2)