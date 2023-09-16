#!/usr/bin/env bash

python -m direct_recovery.direct_recovery --dataset=cora --algorithm=gcn --k=4
python -m direct_recovery.direct_recovery --dataset=citeseer --algorithm=gcn --k=3
python -m direct_recovery.direct_recovery --dataset=actor --algorithm=gcn --k=4
python -m direct_recovery.direct_recovery --dataset=facebook --algorithm=gcn --k=15