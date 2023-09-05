import os

from dotenv import load_dotenv
from torch_geometric.datasets import Planetoid

load_dotenv()

dataset = Planetoid(root=os.environ["DATASET_DIR"], name="CiteSeer")
breakpoint()
