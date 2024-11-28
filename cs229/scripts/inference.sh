#!/bin/bash

CUDA_VISIBLE_DEVICES=7

NGPUS=1
PYTHONPATH=$(git rev-parse --show-toplevel)

# torchrun --nproc_per_node=1 -m cs229.util --testall_local
torchrun --nproc_per_node=1 -m cs229.util --gendata
