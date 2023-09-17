import torch
import torch.nn.functional as F


def distance(x, dist):
    if dist == "cosine":
        x_norm = F.normalize(x, p=2, dim=1)
        result = torch.mm(x_norm, x_norm.transpose(0, 1))
    return result


def top_k(x, k, dist):
    edge = [[], []]
    num_node = x.shape[0]
    pairwire = distance(x, dist)
    values, indices = torch.topk(pairwire.flatten(), (k * num_node) // 2)

    for ind in indices:
        edge[1].append(ind // num_node)
        edge[0].append(ind % num_node)

    return edge
