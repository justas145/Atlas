import argparse
import os
import sys
from typing import List, Dict
import yaml
import logging
import threading
import time
sys.path.append("src")
# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the bluesky directory
bluesky_dir = os.path.abspath(os.path.join(current_dir, "..", "bluesky"))
# Add the bluesky directory to sys.path
sys.path.insert(0, bluesky_dir)

from utils.file_utils import list_scn_files, load_config
from utils.simulator_interface import initialize_simulator, load_and_run_scenario
from utils.experience_library import initialize_experience_library
from utils.load_apis_and_configs import load_agent_configs, load_groq_api_keys, cyclic_key_generator
from agents import setup_agent
from utils.output_capturing import CaptureAndPrintConsoleOutput, ScreenRecorder
from utils.data_processing import get_num_ac_from_scenario, extract_conflict_type, save_results_to_csv
from utils.text_utils import remove_ansi_escape_sequences
from utils.monitoring import (
    monitor_crashes,
    monitor_too_many_requests,
    print_output,
    final_check,
)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run ATC conflict resolution simulations.")
    parser.add_argument("--model_name", type=str, default="gpt-4o-2024-08-06", help="Name of the model to use")
    parser.add_argument(
        "--agent_type",
        type=str,
        default="single_agent",
        choices=["single_agent", "multi_agent"],
        help="Type of agent to use",
    )
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature for the model")
    parser.add_argument("--scenario", type=str, help="Specific scenario to run. If not provided, all scenarios will be run.")
    parser.add_argument("--output_csv", type=str, help="Name of the CSV file to save results. If not provided, results won't be saved.")
    parser.add_argument("--use_skill_lib", action="store_true", help="Whether to use the skill library")
    parser.add_argument(
        "--exp_lib_collection",
        type=str,
        default="experience_library_v3",
        help="Name of the experience library collection to use",
    )
    return parser.parse_args()

def run_simulation(agent_config: Dict, scenario: str, base_path: str, client, collection, groq_api_keys: List[str]) -> Dict:
    # Initialize variables for rerun logic
    rerun_scenario = True
    attempts_count = 0
    max_attempts = 5
    while rerun_scenario and attempts_count < max_attempts:
        attempts_count += 1

        num_ac = get_num_ac_from_scenario(os.path.join(base_path, scenario))
        conflict_type = extract_conflict_type(scenario)
        conflict_with_dH = 'dH' in scenario
        scenario_name = "_".join(scenario.split("/")[-3:]).replace(".scn", "")

        success = False
        for attempt in range(5):
            print(scenario)# Retry up to 5 times
            load_and_run_scenario(client, scenario)
            try:
                with CaptureAndPrintConsoleOutput() as output:
                    # Use the current API key to setup the agent
                    agent_executor = setup_agent(agent_config, groq_api_keys, client, collection)
                    start_time = (
                        time.perf_counter()
                    )  # Record the start time with higher precision
                    result = agent_executor.invoke(
                        {
                            "input": "You are an air traffic controller with tools. Solve aircraft conflict. Solve until there are no more conflicts."
                        }
                    )
                    elapsed_time = (
                        time.perf_counter() - start_time
                    )  # Calculate the elapsed time with higher precision
                success = True
                break  # Exit the retry loop if successful
            except Exception as e:
                print(f"Attempt {attempt + 1} failed, error: {e}")
                if attempt < 4:  # Only sleep and swap keys if it's not the last attempt
                    time.sleep(5)
                    # Swap to the next API key

        if not success:
            print(f"Skipping scenario {scenario_name} after 5 failed attempts.")
            console_output = 'skipped'  # Skip to the next scenario
            elapsed_time = -1  # Mark the scenario as skipped
        else:
            console_output = output.getvalue()
            console_output = remove_ansi_escape_sequences(console_output)

        rerun_scenario = "tool-use>" in console_output or console_output.count(
            "Invoking: `SENDCOMMAND`") == 0
        if rerun_scenario:
            print('reruning the scenario, because of TOOL ERROR')     

    final_score, final_details = final_check()
    print("FINAL SCORE:", final_score)

    num_send_commands = console_output.count("Invoking: `SENDCOMMAND`")
    num_total_commands = console_output.count("Invoking:")

    return {
        "scenario": scenario_name,
        "num_aircraft": num_ac,
        "conflict_type": conflict_type,
        "conflict_with_dH": conflict_with_dH,
        "agent_type": agent_config["type"],
        "model_name": agent_config["model_name"],
        "temperature": agent_config["temperature"],
        "runtime": elapsed_time,  # You need to implement timing logic
        "num_total_commands": num_total_commands,
        "num_send_commands": num_send_commands,
        "score": final_score,
        "log": console_output,
        "final_details": final_details,
        "json_results": result,
        "experience_library": agent_config.get("use_skill_lib", False),
    }

def main():
    # Run the crash monitoring in a background thread
    threading.Thread(target=monitor_crashes, args=(print_output,), daemon=True).start()
    # monitor too many request -> pause simulator in the background
    threading.Thread(target=monitor_too_many_requests, daemon=True).start()
    args = parse_arguments()

    # Define base paths
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bluesky", "scenario"))
    target_path = os.path.join(base_path, "TEST", "Big")

    # Ensure the paths exist
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base scenario path not found: {base_path}")
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Target scenario path not found: {target_path}")

    # Log the paths for debugging
    logging.info(f"Base scenario path: {base_path}")
    logging.info(f"Target scenario path: {target_path}")

    collection = initialize_experience_library(args.exp_lib_collection)
    client = initialize_simulator()

    if args.scenario:
        scenarios = [args.scenario]
    else:
        scenarios = list_scn_files(base_path, target_path)

    agent_config = {
        "type": args.agent_type,
        "model_name": args.model_name,
        "temperature": args.temperature,
        "use_skill_lib": args.use_skill_lib
    }

    groq_api_keys = load_groq_api_keys()
    groq_key_generator = cyclic_key_generator(groq_api_keys)

    results = []
    for scenario in scenarios:
        result = run_simulation(agent_config, scenario, base_path, client, collection, groq_api_keys)
        results.append(result)

        if args.output_csv:
            save_results_to_csv([result], args.output_csv)

    print(f"Completed {len(results)} simulations.")

if __name__ == "__main__":
    main()
