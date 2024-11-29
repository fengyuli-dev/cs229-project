#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

torchrun --nproc_per_node=1 -m cs229.util --testall_local
# torchrun --nproc_per_node=1 -m cs229.util --gendata
