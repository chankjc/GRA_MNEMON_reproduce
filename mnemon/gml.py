import argparse
import os
import os.path as osp
import time

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from dotenv import load_dotenv
from torch_geometric.datasets import Actor, FacebookPagePage, Planetoid
from torch_geometric.nn import GAE, VGAE, GCNConv
from torch_geometric.utils import negative_sampling
import mnemon.gumble_sampling as gs

load_dotenv()

class GML(torch.nn.Module):
    def __init__(self, embedding_size, m):
        super().__init__()
        self.embedding_size = embedding_size
        self.m = m
        self.multihead  = nn.Parameter(torch.randn(m, embedding_size, dtype = torch.float32))
    
    def forward(self, edges, embedding):
        negative_edges = negative_sampling(edges)
        dist = []
        for i in range(self.m):
            x = torch.mul(self.multihead[i], embedding)
            x_norm = F.normalize(x, p=2, dim=1)
            cosine_similarity = torch.mm(x_norm, x_norm.transpose(0, 1))
            cosine_distance = 1 - cosine_similarity
            dist.append( (cosine_distance + 1e-6) /self.m)

        dist = torch.stack(dist, dim=0).sum(dim=0)
        loss_p = torch.stack([torch.log(dist[edges[0][i]][edges[1][i]]) for i in range(len(edges[0]))]).sum(0)
        loss_n = torch.stack([torch.log(1 -  dist[negative_edges[0][i]][negative_edges[1][i]] / 2) for i in range(len(negative_edges[0]))]).sum(0)
        loss = -1 * (loss_p + loss_n)
        
        return loss
    
    def reconstruct(self, embedding, temperature, k):
        dist = []
        for i in range(self.m):
            norm = F.normalize(torch.mul(self.multihead[i], embedding), p=2, dim=1)
            dist.append( (1 - torch.mm(norm, norm.T ) ) /self.m )
        dist = torch.stack(dist, dim=0).sum(dim=0)
        reconstruct_edges = gs.gumble_sampling(dist, temperature, k + 1, distance = "none")
        return reconstruct_edges


def load_GML_model(embedding_size, m = 16):
    model = GML(embedding_size, m)
    return model