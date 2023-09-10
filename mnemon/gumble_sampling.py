import numpy
import torch
from torch import nn
from torch.nn import Module, ModuleList, Sequential


#Euclidean distance
def pairwise_euclidean_distances(x, dim=-1):
    dist = torch.cdist(x,x)**2
    return dist, x


def gumble_sampling(x, t, k, distance = "euclidean"):
    if distance == "euclidean":
        logits, _x = pairwise_euclidean_distances(x)


    temperature = nn.Parameter(torch.tensor(t).float())
    b,n = logits.shape 
    logits = logits * torch.exp(torch.clamp(temperature,-5,5))
        
    q = torch.rand_like(logits) + 1e-8
    lq = (logits-torch.log(-torch.log(q)))
    logprobs, indices = torch.topk(-lq, k + 1)  # 10 x k
    
    A = torch.zeros(b, n)
    for i in range(len(indices)):
        for j in indices[i]:
            if i != j:
                A[i][j] = 1
    return A