#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5,6,7

NGPUS=4
PYTHONPATH=$(git rev-parse --show-toplevel)

torchrun --nproc_per_node=1 -m cs229.inference_local
