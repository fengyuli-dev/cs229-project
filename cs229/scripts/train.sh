#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5,6,7

# Clear prev output
rm -rf output
mkdir output

tune run --nproc_per_node 4 -m cs229.full_finetune_distributed --config configs/config.yaml
