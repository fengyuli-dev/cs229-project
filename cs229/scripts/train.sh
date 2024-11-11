#!/bin/bash

CUDA_VISIBLE_DEVICES=7

# Clear prev output
rm -rf output
mkdir output

tune run --nproc_per_node 1 -m cs229.full_finetune_distributed --config /lfs/local/0/fengyuli/cs229-project/cs229/configs/config.yaml
