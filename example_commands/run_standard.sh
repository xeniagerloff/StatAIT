#!/bin/bash

data_dir="/data"
output_dir="/output"
num_known="300"

python -m stat_ait.run_standard \
    --data_dir "${data_dir}" \
    --output_dir "${output_dir}" \
    --num_known "${num_known}" \
    --seed 12 
