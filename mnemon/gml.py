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

def load_GML_model(embedding_size, m = 16):
    if not variational and not linear:
        model = GAE(GCNEncoder(in_channels, out_channels), InnerProductDecoder())
    elif not variational and linear:
        model = GAE(LinearEncoder(in_channels, out_channels), InnerProductDecoder())
    elif variational and not linear:
        model = VGAE(
            VariationalGCNEncoder(in_channels, out_channels), InnerProductDecoder()
        )
    elif variational and linear:
        model = VGAE(
            VariationalLinearEncoder(in_channels, out_channels), InnerProductDecoder()
        )
    return model