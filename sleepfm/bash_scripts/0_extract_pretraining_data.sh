#!/bin/bash

# Activate the conda environment 
source activate sleepfm_env

num_files=-1
chunk_duration=30.0
num_threads=4

python3 ../0_extract_pretraining_data.py --num_files $num_files \
                                     --chunk_duration $chunk_duration \
                                     --num_threads $num_threads