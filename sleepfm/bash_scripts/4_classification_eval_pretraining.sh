#!/bin/bash

# Activate the conda environment 
source activate sleepfm_env

num_per_event=-1
# model_name="xgb"
model_name="logistic"
max_iter=1000

output_files=("output_leave_one_out_dataset_events_-1_lr_0.001_lr_sp_5_wd_0.0_bs_32_respiratory_sleep_stages_ekg"
             "output_pairwise_dataset_events_-1_lr_0.001_lr_sp_5_wd_0.0_bs_32_respiratory_sleep_stages_ekg"
             )

modality_types=("combined")
for output_file in "${output_files[@]}"; do
    for modality_type in "${modality_types[@]}"; do
        python ../4_classification_eval_pretraining.py \
                --output_file "$output_file" \
                --num_per_event $num_per_event \
                --max_iter $max_iter \
                --modality_type $modality_type \
                --model_name $model_name
    done
done