import argparse
import os
import os.path as osp
import time

import torch
import torch_geometric.transforms as T
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
from torch_geometric.datasets import Actor, FacebookPagePage, Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling

load_dotenv()
from dgl.nn.pytorch.factory import KNNGraph
from metric.confusion_matrix import confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="cora",
    choices=["cora", "citeseer", "actor", "facebook"],
)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--k", type=int, default=8)
parser.add_argument("--distance", type=str, default="cosine")
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
embeddings = {}
datasets["cora"] = Planetoid(
    root=os.environ["DATASET_DIR"], name="Cora", transform=transform
)
embeddings["cora"] = torch.load(f"{os.environ['EMBEDDING_DIR']}gcn/cora/data.pt")
datasets["citeseer"] = Planetoid(
    root=os.environ["DATASET_DIR"], name="CiteSeer", transform=transform
)
embeddings["citeseer"] = torch.load(f"{os.environ['EMBEDDING_DIR']}gcn/citeseer/data.pt")
datasets["actor"] = Actor(
    root=os.environ["DATASET_DIR"] + "/Actor", transform=transform
)
embeddings["actor"] = torch.load(f"{os.environ['EMBEDDING_DIR']}gcn/actor/data.pt")
datasets["facebook"] = FacebookPagePage(
    root=os.environ["DATASET_DIR"] + "/Facebook", transform=transform
)
embeddings["facebook"] = torch.load(f"{os.environ['EMBEDDING_DIR']}gcn/facebook/data.pt")
dataset = datasets[args.dataset]
embedding = embeddings[args.dataset]
num_nodes, num_features = dataset.x.shape
real_edges = dataset.edge_index

knng = KNNGraph(args.k+1)
reconstruct_graph = knng(dataset.x, dist=args.distance)
reconstruct_edge = reconstruct_graph.edges()
precision, recall, f1_score = confusion_matrix(reconstruct_edge, real_edges)
