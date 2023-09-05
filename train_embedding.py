import os

from dotenv import load_dotenv
from torch_geometric.datasets import Actor, FacebookPagePage, Planetoid

load_dotenv()


def prepare_dataset():
    dataset = {}
    dataset["cora"] = Planetoid(root=os.environ["DATASET_DIR"], name="Cora")
    dataset["citeseer"] = Planetoid(root=os.environ["DATASET_DIR"], name="CiteSeer")
    dataset["actor"] = Actor(root=os.environ["DATASET_DIR"])
    dataset["facebook"] = FacebookPagePage(root=os.environ["DATASET_DIR"])
    return dataset


def main():
    dataset = prepare_dataset()
    print(dataset["cora"].x)


if __name__ == "__main__":
    main()
