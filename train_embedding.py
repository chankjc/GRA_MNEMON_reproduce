import os

from dotenv import load_dotenv
from torch_geometric.datasets import Actor, Planetoid

load_dotenv()


def prepare_dataset():
    dataset = {}
    dataset["cora"] = Planetoid(root=os.environ["DATASET_DIR"], name="Cora")
    dataset["citeseer"] = Planetoid(root=os.environ["DATASET_DIR"], name="CiteSeer")
    dataset["actor"] = Actor(root=os.environ["DATASET_DIR"])
    return dataset


def main():
    dataset = prepare_dataset()
    print(len(dataset))


if __name__ == "__main__":
    main()
