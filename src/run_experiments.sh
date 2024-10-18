#!/bin/bash

# Define arrays for agent types, skill library usage, and scenarios
agent_types=(
    single_agent 
)
use_skill_lib=(
    no
)
scenarios=(
    TEST/Big/ac_3/dH/head-on_10.scn

)

# model_names=(
#     "llama3-70b-8192"
# )

# Iterate through each scenario
for i in "${!scenarios[@]}"
do
    PYTHONDONTWRITEBYTECODE=1 python conflict_main.py \
    --agent_type "${agent_types[$i]}" \
    --use_skill_lib "${use_skill_lib[$i]}" \
    --scenario "${scenarios[$i]}" \
    --output_csv results/good_configs_rerun_copy.csv
done
