import argparse
import os
import os.path as osp
import time

import torch
from torch import Tensor
import torch_geometric.transforms as T
from dotenv import load_dotenv
from torch_geometric.datasets import Actor, FacebookPagePage, Planetoid
from torch_geometric.nn import GAE, VGAE, GCNConv

load_dotenv()

# reference: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html
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
    
class InnerProductDecoder(torch.nn.Module):
    def forward(self, z: Tensor, sigmoid: bool = True):
        value = torch.mm(z, z.transpose(0, 1))
        return torch.sigmoid(value) if sigmoid else value
    
def load_GAE_model(in_channels, out_channels, variational = False, linear = False):
    if not variational and not linear:
        model = GAE(GCNEncoder(in_channels, out_channels), InnerProductDecoder())
    elif not variational and linear:
        model = GAE(LinearEncoder(in_channels, out_channels), InnerProductDecoder())
    elif variational and not linear:
        model = VGAE(VariationalGCNEncoder(in_channels, out_channels), InnerProductDecoder())
    elif variational and linear:
        model = VGAE(VariationalLinearEncoder(in_channels, out_channels), InnerProductDecoder())
    return model

def main():
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

if __name__ == '__main__':
    main()
