import argparse
import os
import os.path as osp
import sys

import torch
import torch_geometric.transforms as T
from dotenv import load_dotenv
from sklearn.manifold import TSNE
from torch.nn import Embedding
from torch.utils.data import DataLoader
from torch_geometric.datasets import Actor, FacebookPagePage, Planetoid
from torch_geometric.nn import Node2Vec

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="cora",
    choices=["cora", "citeseer", "actor", "facebook"],
)
parser.add_argument("--epochs", type=int, default=400)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--p", type=float, default=1)
parser.add_argument("--q", type=float, default=1)
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
            add_negative_train_samples=False,
        ),
    ]
)

datasets = {}
datasets["cora"] = Planetoid(
    root=os.environ["DATASET_DIR"], name="Cora", transform=transform
)
datasets["citeseer"] = Planetoid(
    root=os.environ["DATASET_DIR"], name="CiteSeer", transform=transform
)
datasets["actor"] = Actor(
    root=os.environ["DATASET_DIR"] + "/Actor", transform=transform
)
datasets["facebook"] = FacebookPagePage(
    root=os.environ["DATASET_DIR"] + "/Facebook", transform=transform
)
dataset = datasets[args.dataset]
data = dataset

model = Node2Vec(
    data.edge_index,
    embedding_dim=128,
    walk_length=20,
    context_size=10,
    walks_per_node=10,
    num_negative_samples=1,
    p=args.p,
    q=args.q,
    sparse=True,
).to(device)

num_workers = 0 if sys.platform.startswith('win') else 4
loader = model.loader(batch_size=128, shuffle=True,
                    num_workers=num_workers)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def test():
    model.eval()
    z = model()
    acc = model.test(z[data.train_mask], data.y[data.train_mask],
                    z[data.test_mask], data.y[data.test_mask],
                    max_iter=150)
    return acc

@torch.no_grad()
def plot_points(colors):
    model.eval()
    z = model(torch.arange(data.num_nodes, device=device))
    z = TSNE(n_components=2).fit_transform(z.cpu().numpy())
    y = data.y.cpu().numpy()


for epoch in range(1, args.epochs + 1):
    loss = train()
    acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}')

