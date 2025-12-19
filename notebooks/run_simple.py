import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

project_root = Path("/Users/florianwolf/PycharmProjects/graph-nmr2/data/uniform_dist")
sys.path.insert(0, str(project_root))

from dgmc.datasets import PairData, generate_y_t
from dgmc.models import DGMC, GIN, GINE, GINEN, GCN

device = "cuda" if torch.cuda.is_available() else "cpu"

from dgmc.datasets import GraphNmrDataset, pre_transform_to_ones

# Create dataset with pre_transform function to convert x_s and x_t to ones
def prepare_dataloader(folder):
    nmr_dataset = GraphNmrDataset(
        root=project_root / folder,
        # root=project_root / "data/nmr_graphs_1HS7",
        pre_transform=pre_transform_to_ones,
        force_reload=True
    )


    data = nmr_dataset[0]
    dataset = [data] * 16
    dataloader = DataLoader(dataset, batch_size=2, follow_batch=["x_s", "x_t"])
    return data, dataloader





# Run through model
def run():
    test_graph_folders = os.listdir(project_root)
    try:
        for folder in test_graph_folders:
            print(folder)
            data, dataloader = prepare_dataloader(folder)
            torch.manual_seed(2)
            print("initializing DGMC model")
            psi_1 = GINE(in_channels=data.x_s.size(1), out_channels=3, num_layers=2)
            psi_2 = GINE(in_channels=32, out_channels=32, num_layers=2)

            model = DGMC(psi_1, psi_2, num_steps=10).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


            def train():
                total_loss = total_examples = total_correct = total_correct_k = 0
                model.train()
                for batch in dataloader:
                    optimizer.zero_grad()
                    data = batch.to(device)
                    S_0, S_L = model(
                        data.x_s,
                        data.edge_index_s,
                        getattr(data, "edge_attr_s", None),
                        data.x_s_batch,
                        data.x_t,
                        data.edge_index_t,
                        getattr(data, "edge_attr_t", None),
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
                    total_loss / len(dataloader),
                    total_correct / total_examples,
                    total_correct_k / total_examples,
                )

            print("start training")
            accuracy = []
            for epoch in range(1, 20):
                loss, acc, acc_k = train()
                accuracy.append(acc)
                print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.2f}, Acc_k: {acc_k:.2f}")
            plt.plot(range(1, 20), accuracy, label=str(folder))

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc="best")
        plt.savefig("accuracy.pdf", format="pdf")
        plt.show()

    except RuntimeError as e:
        print("RuntimeError: {}".format(e))
    except NotADirectoryError as e:
        print("NotADirectoryError: {}".format(e))


run()
# print(S_L)
# correct = model.acc(S_L, y, reduction="sum")
# print(f"Correct: {correct}")
