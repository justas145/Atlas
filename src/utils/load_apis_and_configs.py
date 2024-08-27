import os
import itertools

def load_agent_configs(config):
    agent_params = config["agents"]
    # Generate all combinations of agent configurations
    all_combinations = list(
        itertools.product(
            agent_params["types"],
            agent_params["model_names"],
            agent_params["temperatures"],
            agent_params["use_skill_libs"],
        )
    )
    # Convert combinations into dictionary format expected by the rest of your code
    expanded_agents = []
    for combo in all_combinations:
        agent_dict = {
            "type": combo[0],
            "model_name": combo[1],
            "temperature": combo[2],
            "use_skill_lib": combo[3],
        }
        expanded_agents.append(agent_dict)
    return expanded_agents


def load_groq_api_keys():
    keys = []
    # First, try to get the first API key without a suffix
    first_key = os.getenv("GROQ_API_KEY")
    if first_key:
        keys.append(first_key)

    # Start from 2 for the subsequent keys
    index = 2
    while True:
        key_name = f"GROQ_API_KEY_{index}"
        key = os.getenv(key_name)
        if not key:
            break
        keys.append(key)
        index += 1
    return keys


def cyclic_key_generator(keys):
    while True:
        for key in keys:
            yield key
