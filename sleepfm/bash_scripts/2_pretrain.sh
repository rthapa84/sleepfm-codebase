#!/bin/bash

# Activate the conda environment 
source activate sleepfm_env

epochs=20
lr=1e-3
lr_step_period=5

# Define modes
modes=("pairwise" "leave_one_out")

# Loop through each mode and execute the python script
for mode in "${modes[@]}"; do
    echo "Running with mode: $mode"
    python3 ../2_pretrain.py \
        --epochs $epochs \
        --mode $mode \
        --batch_size 32 \
        --lr $lr \
        --lr_step_period $lr_step_period
done
