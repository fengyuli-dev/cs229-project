#!/bin/bash

CUDA_VISIBLE_DEVICES=7

# Clear prev output
# rm -rf /lfs/local/0/fengyuli/output/cs229/
# mkdir /lfs/local/0/fengyuli/output/cs229/

tune run --nproc_per_node 1 -m cs229.full_finetune_distributed --config /lfs/local/0/fengyuli/cs229-project/cs229/configs/config_1b.yaml
