import os
import os.path as osp

import torch
import torch_geometric.transforms as T
from dotenv import load_dotenv
from torch_geometric.datasets import Actor, FacebookPagePage, Planetoid


def prepare_dataset(name="cora", algo="gcn", device="cuda:0"):
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
    datasets["cora"] = Planetoid(
        root=os.environ["DATASET_DIR"], name="Cora", transform=transform
    )
    embeddings["cora"] = torch.load(f"{os.environ['EMBEDDING_DIR']}{algo}/cora/data.pt")
    datasets["citeseer"] = Planetoid(
        root=os.environ["DATASET_DIR"], name="CiteSeer", transform=transform
    )
    embeddings["citeseer"] = torch.load(
        f"{os.environ['EMBEDDING_DIR']}{algo}/citeseer/data.pt"
    )
    datasets["actor"] = Actor(
        root=os.environ["DATASET_DIR"] + "/Actor", transform=transform
    )
    embeddings["actor"] = torch.load(
        f"{os.environ['EMBEDDING_DIR']}{algo}/actor/data.pt"
    )
    datasets["facebook"] = FacebookPagePage(
        root=os.environ["DATASET_DIR"] + "/Facebook", transform=transform
    )
    embeddings["facebook"] = torch.load(
        f"{os.environ['EMBEDDING_DIR']}{algo}/facebook/data.pt"
    )

    return datasets[name], embeddings[name], datasets[name].x, datasets[name].edge_index
