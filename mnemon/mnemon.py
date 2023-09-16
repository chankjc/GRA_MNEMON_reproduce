import argparse
import os
import os.path as osp
import time

import torch
import torch_geometric.transforms as T
from dotenv import load_dotenv
from metric.confusion_matrix import confusion_matrix
from torch_geometric.datasets import Actor, FacebookPagePage, Planetoid
from torch_geometric.nn import GAE, VGAE, GCNConv

import mnemon.gumble_sampling as gs
import mnemon.loss

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--variational", action="store_true")
parser.add_argument("--linear", action="store_true")
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
parser.add_argument("--epochs", type=int, default=400)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--learn_rate", type=float, default=0.01)
parser.add_argument("--temperature", type=float, default=4.0)
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
x = dataset.x
real_edges = dataset.edge_index

# first step: init
init_reconstruct_edges = gs.gumble_sampling(dataset.x, args.temperature, args.k + 1)
init_reconstruct_edges = torch.tensor(init_reconstruct_edges).to(device)
print(args)
print("\n")
print("first step:")
precision, recall, f1_score = confusion_matrix(init_reconstruct_edges, real_edges)
exit()

#second step: GML
# third step: GAE
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class VariationalLinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_mu = GCNConv(in_channels, out_channels)
        self.conv_logstd = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

in_channels, out_channels = dataset.num_features, 16

if not args.variational and not args.linear:
    model = GAE(GCNEncoder(in_channels, out_channels))
elif not args.variational and args.linear:
    model = GAE(LinearEncoder(in_channels, out_channels))
elif args.variational and not args.linear:
    model = VGAE(VariationalGCNEncoder(in_channels, out_channels))
elif args.variational and args.linear:
    model = VGAE(VariationalLinearEncoder(in_channels, out_channels))

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    '''
        todo
    '''
    loss = model.recon_loss(z, train_data.pos_edge_label_index)
    if args.variational:
        loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)


times = []
for epoch in range(1, args.epochs + 1):
    start = time.time()
    loss = train()
    auc, ap = test(test_data)
    print(f"Epoch: {epoch:03d}, AUC: {auc:.4f}, AP: {ap:.4f}")
    times.append(time.time() - start)

print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")



# torch.save(init_g, f"{os.environ['RESULT_DIR']}gcn/{args.dataset}/graph.pt")