#!/usr/bin/env bash

python mnemon.py --dataset=cora --algorithm=gcn --k=8
python mnemon.py --dataset=citeseer --algorithm=gcn --k=5 
python mnemon.py --dataset=actor --algorithm=gcn --k=8
python mnemon.py --dataset=facebook --algorithm=gcn --k=30
