#!/usr/bin/env bash

python -m knn.knn --dataset=cora --algorithm=gcn --k=4
python -m knn.knn --dataset=citeseer --algorithm=gcn --k=3
python -m knn.knn --dataset=actor --algorithm=gcn --k=4
python -m knn.knn --dataset=facebook --algorithm=gcn --k=15