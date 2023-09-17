import argparse
import copy
import os
import os.path as osp
import time

import torch
import torch_geometric.transforms as T
from dotenv import load_dotenv
from torch_geometric.datasets import Actor, FacebookPagePage, Planetoid
from torch_geometric.nn import GAE, VGAE, GCNConv
from torch_geometric.utils import to_scipy_sparse_matrix
from tqdm import tqdm

import mnemon.gumble_sampling as gs
import mnemon.reconstruct_loss as rl
from metric.confusion_matrix import confusion_matrix
from mnemon.gae import *
from mnemon.gml import *
from mnemon.prepare_dataset import prepare_dataset

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
parser.add_argument("--gae_epochs", type=int, default=10000)
parser.add_argument("--gml_epochs", type=int, default=10000)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--learn_rate", type=float, default=0.01)
parser.add_argument("--temperature", type=float, default=4.0)
parser.add_argument("--k", type=int, default=4)
parser.add_argument("--distance", type=str, default="cosine")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--without_gml", action="store_true")
parser.add_argument("--alpha", type=float, default=0.3)
parser.add_argument("--beta", type=float, default=0.1)
parser.add_argument("--eta", type=float, default=0.5)
parser.add_argument("--round", type=int, default=10)
parser.add_argument("--threadhold", type=float, default=0.7)
parser.add_argument("--m", type=int, default=16)
args = parser.parse_args()

torch.manual_seed(args.seed)

if torch.cuda.is_available():
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.manual_seed_all(args.seed)
else:
    device = torch.device("cpu")


def main():
    print("MNEMON method:")
    print(args)

    print("================")
    print("| Load dataset |")
    print("================")
    print(f"> dataset: {args.dataset}\n")
    dataset, embedding, x, real_edges = prepare_dataset(
        args.dataset, args.algorithm, device=device
    )
    num_node, embedding_dim = embedding.shape
    
    real_embeddind = copy.deepcopy(embedding)

    # first step: init
    print("===============================")
    print("| first step: Gumble sampling |")
    print("===============================")
    initial_edges = gs.gumble_sampling(dataset.x, args.temperature, args.k + 1)
    initial_edges = torch.tensor(initial_edges).to(device)
    precision, recall, f1_score = confusion_matrix(initial_edges, real_edges)
    init_adj = to_scipy_sparse_matrix(initial_edges)
    init_adj = init_adj.toarray()
    init_adj = torch.from_numpy(init_adj).to(device)

    reconstruct_edges = copy.deepcopy(initial_edges)
    new_adj = copy.deepcopy(init_adj)

    for round in tqdm(range(1, args.round + 1)):
        tqdm.write("==============")
        tqdm.write(f"> round: {round}")
        tqdm.write("==============")
        # second step: GML
        tqdm.write("======================================")
        tqdm.write("| second step: Graph Metric Learning |")
        tqdm.write("======================================")
        if not args.without_gml:
            model_gml = load_GML_model(embedding_dim, args.m)
            optimizer_gml = torch.optim.Adam(model_gml.parameters(), lr=args.learn_rate)
            
            def train_GML(model_gml, optimizer_gml, reconstruct_edges, embedding):
                model_gml.train()

            @torch.no_grad()
            def test_GML(model_gml, embedding):
                model_gml.eval()

            for epoch in tqdm(range(1, args.gml_epochs + 1)):
                loss = train_GML(model_gml, optimizer_gml, reconstruct_edges, embedding)
                if epoch % 50 == 0:
                    tqdm.write(f"GML => Epoch: {epoch:03d}, Loss: {loss}")

            reconstruct_edges = test_GML(model_gml, embedding)
            new_adj = to_scipy_sparse_matrix(reconstruct_edges)
            new_adj = new_adj.toarray()
            new_adj = torch.from_numpy(new_adj).to(device)

            precision, recall, f1_score = confusion_matrix(reconstruct_edges, real_edges)

        else:
            tqdm.write("> w/o GML!")

        # third step: GAE
        # reference: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/autoencoder.html
        tqdm.write("=================================")
        tqdm.write("| third step: Graph AutoEncoder |")
        tqdm.write("=================================")

        in_channels, out_channels = embedding.shape[1], embedding.shape[1] * 2

        model_gae = load_GAE_model(
            in_channels, out_channels, variational=args.variational, linear=args.linear
        )
        model_gae = model_gae.to(device)
        optimizer_gae = torch.optim.Adam(model_gae.parameters(), lr=args.learn_rate)

        def train_GAE(model_gae, optimizer_gae, adj, reconstruct_edges, embedding, real_embeddind):
            model_gae.train()
            optimizer_gae.zero_grad()
            z = model_gae.encode(embedding, reconstruct_edges)
            new_adj = model_gae.decode(z)

            loss = rl.graph_laplacian_regularization(new_adj, real_embeddind)
            # loss = rl.graph_sparsity_regularization(new_adj, args.alpha, args.beta, args.device)
            # loss = rl.graph_reconstruction_loss(new_adj, adj)
            '''
            loss = (
                rl.graph_laplacian_regularization(new_adj, real_embeddind)
                + rl.graph_sparsity_regularization(
                    new_adj, args.alpha, args.beta, args.device
                )
                + rl.graph_reconstruction_loss(new_adj, adj)
            )
            '''
            if args.variational:
                loss = loss + (1 / num_node) * model_gae.kl_loss()
            loss.backward()
            optimizer_gae.step()
            return float(loss)

        @torch.no_grad()
        def test_GAE(model_gae, reconstruct_edges, embedding):
            model_gae.eval()
            z = model_gae.encode(embedding, reconstruct_edges)
            new_adj = model_gae.decode(z)
            return new_adj, z

        times = []
        for epoch in tqdm(range(1, args.gae_epochs + 1)):
            start = time.time()
            loss = train_GAE(model_gae, optimizer_gae, new_adj, reconstruct_edges, embedding, real_embeddind)
            if epoch % 50 == 0:
                tqdm.write(f"GAE => Epoch: {epoch:03d}, Loss: {loss}")

        new_adj, embedding = test_GAE(model_gae, reconstruct_edges, embedding)
        new_adj = (1 - args.eta) * init_adj + args.eta * new_adj
        new_adj = torch.clamp(new_adj, 0, 1)
        threadhold = args.threadhold
        new_adj = torch.where(
            new_adj < threadhold,
            torch.tensor(0.0),
            torch.where(new_adj >= threadhold, torch.tensor(1.0), new_adj),
        )

        reconstruct_edges = new_adj.nonzero().t().contiguous()
        precision, recall, f1_score = confusion_matrix(reconstruct_edges, real_edges)


if __name__ == "__main__":
    main()
