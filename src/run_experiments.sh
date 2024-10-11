#!/bin/bash

temperatures=(1.2)

for temp in "${temperatures[@]}"
do
    PYTHONDONTWRITEBYTECODE=1 python conflict_main.py \
    --model_name mixtral-8x7b-32768,llama3-8b-8192,gpt-4o-2024-08-06,llama3-70b-8192,gemma2-9b-it \
    --temperature $temp \
    --scenario TEST/Small/ac_4 \
    --output_csv results/test_small_all_models.csv
done