#!/bin/bash

# Activate the conda environment 
source activate sleepfm_env

splits="train,valid,test"

# path to the models pretrained
output_dirs=("outputs/output_leave_one_out_dataset_events_-1_lr_0.001_lr_sp_5_wd_0.0_bs_32_respiratory_sleep_stages_ekg"
             "outputs/output_pairwise_dataset_events_-1_lr_0.001_lr_sp_5_wd_0.0_bs_32_respiratory_sleep_stages_ekg")


for i in "${!output_dirs[@]}"; do
  output_dir="${output_dirs[$i]}"

  python3 ../3_generate_embed_pretraining.py $output_dir --splits $splits
done