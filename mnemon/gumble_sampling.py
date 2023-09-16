import numpy
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module, ModuleList, Sequential


# Euclidean distance
def pairwise_euclidean_distances(x):
    dist = torch.cdist(x,x)**2
    return dist, x

# Cosine distance
def pairwise_cosine_distances(x):
    x_norm = F.normalize(x, p=2, dim=1)
    dist = torch.mm(x_norm, x_norm.transpose(0, 1))
    dist = 1 - dist
    return dist, x

def gumble_sampling(x, t, k, distance = "cosine"):
    if distance == "euclidean":
        logits, _x = pairwise_euclidean_distances(x)

    if distance == "cosine":
        logits, _x = pairwise_cosine_distances(x)
    
    temperature = nn.Parameter(torch.tensor(t).float())
    b,n = logits.shape 
    logits = logits * torch.exp(torch.clamp(temperature,-5,5))
        
    q = torch.rand_like(logits) + 1e-8
    
    lq = (logits-torch.log(-torch.log(q)))
    logprobs, indices = torch.topk(-lq, k + 1)  # 10 x k
    
    recover_edge = [[],[]]
    
    for i in range(b):
        for j in indices[i]:
            recover_edge[1].append(int(i))
            recover_edge[0].append(int(j))

    return recover_edge