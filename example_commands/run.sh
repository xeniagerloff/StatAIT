#!/bin/bash
data_dir="/data"
output_dir="/output"

python -m stat_ait.run \
    --data_dir "${data_dir}" \
    --output_dir "${output_dir}" \
    --seed 12 \
    > "stat_ait.log"