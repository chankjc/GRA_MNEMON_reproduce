import argparse
import os
import os.path as osp
import time

import torch
import torch_geometric.transforms as T
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix
from torch_geometric.datasets import Actor, FacebookPagePage, Planetoid

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="cora",
    choices=["cora", "citeseer", "actor", "facebook"],
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="gcn",
    choices=["cora", "citeseer", "actor", "facebook"],
)
parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device(f"cuda:{args.device}")
else:
    device = torch.device("cpu")

transform = T.Compose(
    [
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(
            num_val=0.05,
            num_test=0.1,
            is_undirected=True,
            split_labels=True,
            add_negative_train_samples=False,
        ),
    ]
)

datasets = {}
graphs = {}
datasets["cora"] = Planetoid(
    root=os.environ["DATASET_DIR"], name="Cora", transform=transform
)
graphs["cora"] = torch.load(f"{os.environ['RESULT_DIR']}gcn/cora/graph.pt")

dataset = datasets[args.dataset]
n = len(dataset.y)
reference_graph = torch.zeros(n,n)
for ind in range(len(dataset.edge_index[0])):
    i = dataset.edge_index[0][ind]
    j = dataset.edge_index[1][ind]
    reference_graph[i][j] = 1

r1 = torch.flatten(reference_graph)
graph = graphs[args.dataset]
g1 = torch.flatten(graph)
breakpoint()
tn, fp, fn, tp = confusion_matrix(r1, g1).ravel()
print("prec:", tp/(tp + fp))
print("recall:", tp/(tp + fn))
breakpoint()