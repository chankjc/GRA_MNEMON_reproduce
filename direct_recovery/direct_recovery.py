import argparse
import heapq
import os
import os.path as osp
import time

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
from scipy import spatial
from torch_geometric.datasets import Actor, FacebookPagePage, Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from tqdm import tqdm

load_dotenv()
from metric.confusion_matrix import confusion_matrix
from direct_recovery.top_k import top_k


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
    choices=["gcn"],
)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--k", type=int, default=8)
parser.add_argument("--distance", type=str, default="cosine")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

torch.manual_seed(args.seed)

if torch.cuda.is_available():
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.manual_seed_all(args.seed)
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
print("direct recover method:")
print(args)

datasets = {}
embeddings = {}
algo = args.algorithm
datasets["cora"] = Planetoid(
    root=os.environ["DATASET_DIR"], name="Cora", transform=transform
)
embeddings["cora"] = torch.load(f"{os.environ['EMBEDDING_DIR']}{algo}/cora/data.pt")
datasets["citeseer"] = Planetoid(
    root=os.environ["DATASET_DIR"], name="CiteSeer", transform=transform
)
embeddings["citeseer"] = torch.load(f"{os.environ['EMBEDDING_DIR']}{algo}/citeseer/data.pt")
datasets["actor"] = Actor(
    root=os.environ["DATASET_DIR"] + "/Actor", transform=transform
)
embeddings["actor"] = torch.load(f"{os.environ['EMBEDDING_DIR']}{algo}/actor/data.pt")
datasets["facebook"] = FacebookPagePage(
    root=os.environ["DATASET_DIR"] + "/Facebook", transform=transform
)
embeddings["facebook"] = torch.load(f"{os.environ['EMBEDDING_DIR']}{algo}/facebook/data.pt")
dataset = datasets[args.dataset]
embedding = embeddings[args.dataset]
num_nodes, num_features = dataset.x.shape
real_edges = dataset.edge_index

reconstruct_edge = top_k(dataset.x, args.k+1, dist=args.distance)

precision, recall, f1_score = confusion_matrix(reconstruct_edge, real_edges)

print("\n")