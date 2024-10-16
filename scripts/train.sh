#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3

# Clear prev output
rm -rf output
mkdir output

tune run --nproc_per_node 4 /lfs/local/0/fengyuli/cs229-project/full_finetune_distributed.py --config /lfs/local/0/fengyuli/cs229-project/configs/config.yaml
