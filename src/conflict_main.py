import os
import sys
from typing import List, Dict
import yaml
import logging
import threading
import time
import click
from langchain.schema import AgentAction, AgentFinish
from langchain_core.messages import AIMessageChunk
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the bluesky directory
bluesky_dir = os.path.abspath(os.path.join(current_dir, "..", "bluesky"))
# Add the bluesky directory to sys.path
sys.path.insert(0, bluesky_dir)

from utils.file_utils import list_scn_files, load_config
from utils.simulator_interface import initialize_simulator, load_and_run_scenario
from utils.experience_library import initialize_experience_library
from utils.load_apis_and_configs import (
    load_agent_configs,
    load_groq_api_keys,
    cyclic_key_generator,
)
from agents import setup_agent, process_agent_output
from utils.output_capturing import CaptureAndPrintConsoleOutput, ScreenRecorder
from utils.data_processing import (
    get_num_ac_from_scenario,
    extract_conflict_type,
    save_results_to_csv,
)
from utils.text_utils import remove_ansi_escape_sequences
from utils.monitoring import (
    monitor_crashes,
    monitor_too_many_requests,
    print_output,
    final_check,
    check_preference_execution,
)
from voice_assistant.api_key_manager import get_transcription_api_key, get_tts_api_key
from voice_assistant.config import Config
from voice_assistant.utils import delete_file
from voice_assistant.text_to_speech import text_to_speech, reset_last_spoken_text
from voice_assistant.transcription import transcribe_audio
from voice_assistant.audio import record_audio, play_audio
from queue import Queue

from tools.agent_tools import get_tlos_logs, get_command_logs
import json
from pathlib import Path

@click.command()
@click.option("--model_name", default="gpt-4o-2024-08-06", help="single model or comma-separated list")
@click.option("--agent_type", default="single_agent", help="single_agent or multi_agent, or comma-separated combination")
@click.option("--temperature", default=0.3, help="single value or comma-separated list")
@click.option("--use_skill_lib", default="no", help="yes or no, or Comma-separated combination")
@click.option("--scenario", help="Specific scenario or folder to run. If empty, all scenarios.")
@click.option("--output_csv", help="CSV file name. If none, results won't be saved.")
@click.option(
    "--exp_lib_collection",
    default="experience_library_v3",
    help="experience library collection name",
)
@click.option("--voice", type=click.Choice(['2-way', '1-way', 'no']), default='no', help="Voice interaction mode")
@click.option("--preference", type=click.Choice(['HDG', 'ALT', 'tLOS']), default=None, help="Preference for conflict resolution")
@click.option("--tlos_threshold", type=float, default=None, help="tLOS threshold for preference (only applicable when preference is 'tLOS')")
def main(
    model_name,
    agent_type,
    temperature,
    use_skill_lib,
    scenario,
    output_csv,
    exp_lib_collection,
    voice,
    preference,
    tlos_threshold,
):
    collection = initialize_experience_library(exp_lib_collection)
    print("collection initialized")
    client = initialize_simulator()
    print("bluesky client initialized")
    # Run the crash monitoring in a background thread
    threading.Thread(target=monitor_crashes, args=(print_output,), daemon=True).start()
    # monitor too many request -> pause simulator in the background
    threading.Thread(target=monitor_too_many_requests, args=(client,), daemon=True).start()
    agent_configs = create_agent_configs(
        model_name, agent_type, temperature, use_skill_lib
    )
    print('monitoring for crashes and too many requests')

    base_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "bluesky", "scenario")
    )
    
    if scenario:
        scenario_path = Path(os.path.join(base_path, scenario))
        if scenario_path.is_dir():
            target_path = scenario_path
            scenarios = list_scn_files(base_path, str(target_path))
        elif scenario.endswith('.scn'):
            scenarios = [scenario]
        else:
            raise ValueError(f"Invalid scenario input: {scenario}")
    else:
        target_path = os.path.join(base_path, "TEST", "Big")
        scenarios = list_scn_files(base_path, target_path)

    print(f"Running scenarios: {scenarios}")

    groq_api_keys = load_groq_api_keys()
    groq_key_generator = cyclic_key_generator(groq_api_keys)

    results = []
    for scenario in scenarios:
        for agent_config in agent_configs:
            result = run_simulation(
                agent_config, scenario, base_path, client, collection, groq_key_generator, voice, preference, tlos_threshold
            )
            results.append(result)

            if output_csv:
                save_results_to_csv([result], output_csv)

    print(f"Completed {len(results)} simulations.")


def create_agent_configs(model_name, agent_type, temperature, use_skill_lib):
    model_names = model_name.split(",")
    agent_types = agent_type.split(",") if "," in agent_type else [agent_type] * len(model_names)
    
    # Convert temperature to string if it's a float
    temperature = str(temperature)
    temperatures = temperature.split(",") if "," in temperature else [temperature] * len(model_names)
    
    use_skill_libs = use_skill_lib.split(",") if "," in use_skill_lib else [use_skill_lib] * len(model_names)

    # Ensure all lists have the same length
    max_length = max(len(model_names), len(agent_types), len(temperatures), len(use_skill_libs))
    model_names = (model_names * max_length)[:max_length]
    agent_types = (agent_types * max_length)[:max_length]
    temperatures = (temperatures * max_length)[:max_length]
    use_skill_libs = (use_skill_libs * max_length)[:max_length]

    agent_configs = []
    for model, agent_type, temp, skill_lib in zip(model_names, agent_types, temperatures, use_skill_libs):
        agent_configs.append({
            "type": agent_type,
            "model_name": model,
            "temperature": float(temp),
            "use_skill_lib": skill_lib.lower() == "yes",
        })
    return agent_configs


def run_simulation(
    agent_config: Dict,
    scenario: str,
    base_path: str,
    client,
    collection,
    groq_key_generator,
    voice_mode: str,
    preference: str,
    tlos_threshold: float,
) -> Dict:
    # Reset the last spoken text at the start of each simulation run
    reset_last_spoken_text()

    # Initialize variables for rerun logic
    rerun_scenario = True
    attempts_count = 0
    max_attempts = 5
    while rerun_scenario and attempts_count < max_attempts:
        attempts_count += 1

        # Reset the last spoken text for each attempt, it is for voice agent
        reset_last_spoken_text()
        scenario = os.path.normpath(scenario)

        num_ac = get_num_ac_from_scenario(os.path.join(base_path, scenario))
        conflict_type = extract_conflict_type(scenario)
        conflict_with_dH = "dH" in scenario
        scenario_name = "_".join(scenario.split("/")[-3:]).replace(".scn", "")

        success = False
        for attempt in range(5):
            print(scenario)  # Retry up to 5 times
            load_and_run_scenario(client, scenario)
            try:
                with CaptureAndPrintConsoleOutput() as output:
                    # Use the groq_key_generator to setup the agent
                    agent_executor = setup_agent(
                        agent_config, groq_key_generator, client, collection
                    )
                    user_input = """
**Objective**: Monitor the airspace and resolve conflicts between aircraft pairs until there are no more conflicts.

**Guidelines**:
You are allowed to change the aircraft altitude and heading.

 """

                    if voice_mode == '2-way':
                        record_audio("user_input.wav")
                        transcription_api_key = get_transcription_api_key()
                        user_input = transcribe_audio(
                            Config.TRANSCRIPTION_MODEL, transcription_api_key, 'user_input.wav', Config.LOCAL_MODEL_PATH)
                        logging.info("User said: " + user_input)
                        print("User said: " + user_input)

                    start_time = time.perf_counter()

                    # Start the audio player thread if voice mode is enabled
                    if voice_mode in ['1-way', '2-way']:
                        audio_thread = None
                    else:
                        audio_thread = None

                    result = process_agent_output(
                        agent_executor, user_input, voice_mode
                    )

                    elapsed_time = time.perf_counter() - start_time

                success = True
                break  # Exit the retry loop if successful
            except Exception as e:
                print(f"Attempt {attempt + 1} failed, error: {e}")
                result = str(e)
                if attempt < 4:  # Only sleep if it's not the last attempt
                    time.sleep(5)

        if not success:
            print(f"Skipping scenario {scenario_name} after 5 failed attempts.")
            console_output = "skipped"  # Skip to the next scenario
            elapsed_time = -1  # Mark the scenario as skipped

        else:
            console_output = output.getvalue()
            console_output = remove_ansi_escape_sequences(console_output)

        # rerun_scenario = (
        #     "tool-use>" in console_output
        #     or console_output.count("Invoking: `SENDCOMMAND`") == 0
        # )
        rerun_scenario = "tool-use>" in console_output

        tlos_logs = get_tlos_logs()
        command_logs = get_command_logs()
        print(tlos_logs)
        print(json.dumps(tlos_logs, indent=2))

        if rerun_scenario:
            print("reruning the scenario, because of TOOL ERROR")

    final_score, final_details = final_check()
    preference_executed = check_preference_execution(
        preference, tlos_threshold, command_logs, tlos_logs
    )
    print("FINAL SCORE:", final_score)
    print("PREFERENCE EXECUTED:", preference_executed)

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
        "preference": preference,
        "preference_executed": preference_executed,
    }

if __name__ == "__main__":
    main()
