#!/bin/bash

# Activate the conda environment 
source activate sleepfm_env

num_threads=4

python3 ../1_prepare_dataset.py \
    --random_state 42 \
    --test_size 100 \
    --num_threads $num_threads \