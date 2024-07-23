import os
import re
import itertools
import colorama
colorama.init(autoreset=True)
import csv
import sys
sys.path.append("../bluesky")
from bluesky.network.client import Client
import dotenv
from agent_tools import (
    initialize_client,
    initialize_collection,
    agent_tools_list,
    GetConflictInfo,
    receive_bluesky_output,
    agent_tool_dict,
)
import time
from langchain import agents, hub
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from contextlib import contextmanager
from prompts.agent_prompts import (
    planner_prompt,
    executor_prompt,
    verifier_prompt,
)

dotenv.load_dotenv("../.env")
# Usage
base_path = "../bluesky/scenario"  # Adjust this to your base directory
target_path = (
    "../bluesky/scenario/TEST"  # Adjust this to the directory you want to search
)

with open("prompts/ICAO_seperation_guidelines.txt", "r") as f:
    icao_seperation_guidelines = f.read()


import os
import cv2
import numpy as np
from PIL import ImageGrab
import pygetwindow as gw


import cv2
import numpy as np
from PIL import ImageGrab
import os
import threading
import time


class ScreenRecorder:
    def __init__(self, output_directory, file_name, fps=20.0):
        self.output_directory = output_directory
        self.file_name = file_name
        self.fps = fps
        self.out = None
        self.is_recording = False
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

    def start_recording(self):
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.screen_size = ImageGrab.grab().size
        full_path = os.path.join(self.output_directory, self.file_name)
        self.out = cv2.VideoWriter(full_path, fourcc, self.fps, self.screen_size)
        self.is_recording = True
        threading.Thread(target=self.record_loop, daemon=True).start()

    def record_loop(self):
        while self.is_recording:
            self.record_frame()
            time.sleep(1 / self.fps)

    def record_frame(self):
        img = ImageGrab.grab()
        img_np = np.array(img)
        frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        self.out.write(frame)

    def stop_recording(self):
        self.is_recording = False
        time.sleep(1 / self.fps)  # wait for the last frame to be processed
        self.out.release()
        cv2.destroyAllWindows()

    def __enter__(self):
        self.start_recording()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_recording()


class CaptureAndPrintConsoleOutput:
    def __init__(self):
        self.old_stdout = sys.stdout
        self.output = []

    def write(self, text):
        self.output.append(text)
        self.old_stdout.write(text)

    def flush(self):
        self.old_stdout.flush()

    def getvalue(self):
        return "".join(self.output)

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout


class MultiAgent:
    def __init__(self, agents, prompts, icao_seperation_guidelines):
        self.agents = agents
        self.prompts = prompts
        self.icao_seperation_guidelines = icao_seperation_guidelines

    def invoke(self, initial_input):
        planner_prompt = self.prompts["planner"].format(
            icao_seperation_guidelines=self.icao_seperation_guidelines
        )
        print("Planner Agent Running")
        plan = self.agents["planner"].invoke({"input": planner_prompt})["output"]

        if "no conflicts" in plan.lower():
            return {"output": plan, "status": "resolved"}

        while True:
            controller_prompt = self.prompts["executor"].format(plan=plan)
            print("Controller Agent Running")
            controller_output = self.agents["controller"].invoke(
                {"input": controller_prompt}
            )["output"]

            verifier_prompt = self.prompts["verifier"].format(
                icao_seperation_guidelines=self.icao_seperation_guidelines, plan=plan
            )
            print("Verifier Agent Running")
            verifier_output = self.agents["verifier"].invoke(
                {"input": verifier_prompt}
            )["output"]
            if "no conflicts" in verifier_output.lower():
                return {"output": verifier_output, "status": "resolved"}
            else:
                plan = verifier_output


def initialize_simulator():
    client = Client()
    client.connect("127.0.0.1", 11000, 11001)  # Adjust IP and ports as necessary
    client.update()
    initialize_client(client)

    return client


# get a list of scn files to load
def list_scn_files(base_path, target_path):
    scn_files = []
    # Convert base_path and target_path to absolute paths to handle relative paths accurately
    base_path = os.path.abspath(base_path)
    target_path = os.path.abspath(target_path)

    # Walk through all directories and files in the target_path
    for root, dirs, files in os.walk(target_path):
        # Filter for .scn files
        for file in files:
            if file.endswith(".scn"):
                # Construct full path to the file
                full_path = os.path.join(root, file)
                # Create a relative path from the base_path
                relative_path = os.path.relpath(full_path, start=base_path)
                # Append the relative path to the list
                relative_path = relative_path.replace('\\', '/')
                scn_files.append(relative_path)
    return scn_files

import yaml


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def load_and_run_scenario(client, scenario_path):
    try:
        clear_crash_log()
        # client.send_event(b"STACK", f"IC {scenario_path}")
        client.send_event(b"STACK", f"IC {scenario_path}")

        client.update()
        print(f"Loaded scenario: {scenario_path}")
        time.sleep(3)  # Wait for the scenario to load
        # clear the output buffer
        out = receive_bluesky_output()

    except Exception as e:
        print(f"Failed to load scenario {scenario_path}: {str(e)}")


def clear_crash_log(crash_log_path="../bluesky/output/crash_log.txt"):
    # Clear the crash log file or create it if it doesn't exist
    with open(crash_log_path, "w") as file:
        file.truncate()


def get_num_ac_from_scenario(scenario_path):
    # Construct the full path to the scenario file
    full_path = os.path.join(base_path, scenario_path)

    try:
        # Open the scenario file and read its content
        with open(full_path, "r") as file:
            content = file.read()

        # Count occurrences of ">CRE" to accurately count aircraft creation commands
        # This accounts for the format where "CRE" follows a timestamp and command prefix
        aircraft_count = content.count(">CRE")

        return aircraft_count
    except FileNotFoundError:
        print(f"File not found: {full_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def try_get_conflict_info(max_attempts=3):
    for attempt in range(max_attempts):
        conflict_info = GetConflictInfo("SHOWTCPA")
        if (
            conflict_info.strip() and conflict_info.strip() != "Error"
        ):  # Assume "Error" is a placeholder for any error message
            return conflict_info  # Return valid data if non-empty and no error
        else:
            print(
                f"Attempt {attempt + 1}: Failed to get valid conflict info. Retrying..."
            )
            time.sleep(2)  # Wait for 2 seconds before retrying

    # If all attempts fail, handle it as needed
    print("Failed to retrieve conflict information after several attempts.")
    return None  # Or any other error handling mechanism


def final_check(crash_log_path="../bluesky/output/crash_log.txt"):
    """
    Performs the final checks after the agent has completed its task.

    Args:
    crash_log_path (str): Path to the crash log file.

    Returns:
    tuple: (score, details) where score is the evaluation score and details contain either crash or conflict information.
    """
    # Check for a crash in the log file
    try:
        with open(crash_log_path, "r") as file:
            crash_info = file.read().strip()
            if crash_info:
                return -1, crash_info  # Crash detected, score -1, return crash details
    except FileNotFoundError:
        print("Crash log file not found. Assuming no crash.")
    except Exception as e:
        print(f"Error reading crash log file: {e}")
        return None

    # Check for conflicts if no crash detected
    conflict_info = try_get_conflict_info()
    if conflict_info is None:
        return 1, "Conflict information could not be retrieved."  
    if conflict_info.strip() != "No conflicts detected.":
        # Parsing the conflict information for DCPA values
        lines = conflict_info.strip().split("\n")[1:]  # Skip the header
        crash_detected = False
        for line in lines:
            parts = line.split("|")
            if len(parts) > 4:  # Ensure there are enough parts to extract DCPA
                dcpa_nmi = float(
                    parts[4].split(":")[1].strip().split(" ")[0]
                )  # Extract DCPA value in nautical miles
                dcpa_meters = dcpa_nmi * 1852  # Convert nautical miles to meters
                if dcpa_meters <= 300:
                    print(f'dcpa is {dcpa_meters}, crash detected!')
                    crash_detected = True
                    break

        if crash_detected:
            return -1, conflict_info  # Crash scenario detected due to DCPA threshold
        else:
            return 0, conflict_info  # Conflicts detected, but no crash
    else:
        return (
            1,
            "No crashes or conflicts detected.",
        )  # No crashes or conflicts, score 1


def extract_conflict_type(file_path):
    # Extract the filename from the full path
    filename = os.path.basename(file_path)

    # Remove the file extension to isolate the conflict type indicator
    conflict_type = os.path.splitext(filename)[0]
    print(conflict_type)
    # Define known conflict types
    known_conflicts = {"converging", "head-on", "parallel", "t-formation"}

    # Replace underscores with hyphens and check against known conflicts
    for known in known_conflicts:
        if known in conflict_type.replace("_", "-"):
            # Special case for 'head_on' to 'head-on'
            return known

    # If no known conflict type is found, return 'undefined'
    return "undefined"


def save_results_to_csv(results, output_file):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Check if the file exists to determine if headers are needed
    file_exists = os.path.isfile(output_file)

    with open(output_file, "a", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "scenario",
            "num_aircraft",
            "conflict_type",
            "agent_type",
            "model_name",
            "temperature",
            "runtime",
            "num_total_commands",
            "num_send_commands",
            "score",
            "log",
            "final_details"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header only if the file did not exist
        if not file_exists:
            writer.writeheader()

        # Write results
        for result in results:
            writer.writerow(result)


def remove_ansi_escape_sequences(text):
    ansi_escape_pattern = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
    return ansi_escape_pattern.sub("", text)

def setup_chat_model(config, groq_api_key):
    if 'gpt' in config["model_name"]:
        chat = ChatOpenAI(temperature=config["temperature"], model_name=config["model_name"])
    else:
        chat = ChatGroq(temperature=config["temperature"], model_name=config["model_name"], api_key=groq_api_key)
    return chat

def setup_single_agent(config, groq_api_key):
    # Load system prompts
    prompt = hub.pull("hwchase17/openai-tools-agent")
    with open("prompts/system.txt", "r") as f:
        prompt.messages[0].prompt.template = f.read()
    print('setting up agent with config:', config)
    # Initialize LLM model
    chat = setup_chat_model(config, groq_api_key)
    # Decide on tools to include based on the 'use_skill_lib' configuration
    if agent_config.get("use_skill_lib", False):
        tools_to_use = (
            agent_tool_dict.values()
        )  # Use all available tools if skill lib is to be used
        print('using skill lib')
    else:
        # Exclude "QueryConflicts" tool if 'use_skill_lib' is False
        tools_to_use = [
            tool for name, tool in agent_tool_dict.items() if name != "QUERYCONFLICTS"
        ]
        print('not using skill lib')
    tools_to_use = list(tools_to_use)
    # Create and return the agent
    agent = agents.create_openai_tools_agent(chat, tools_to_use, prompt)
    agent_executor = agents.AgentExecutor(agent=agent, tools=tools_to_use, verbose=True)
    return agent_executor


def setup_multi_agent(config, groq_api_key):
    print('setting up multi agent with config:', config	)
    agents_dict = {}
    tool_sets = {
        "planner": ["GETALLAIRCRAFTINFO", "GETCONFLICTINFO"],
        "controller": ["SENDCOMMAND"],
        "verifier": ["GETALLAIRCRAFTINFO", "GETCONFLICTINFO", "CONTINUEMONITORING"],
    }

    agents_prompts = {
    'planner': planner_prompt,
    'executor': executor_prompt,
    'verifier': verifier_prompt
    }

    chat = setup_chat_model(config, groq_api_key)
    prompt = hub.pull("hwchase17/openai-tools-agent")

    for role, tool_names in tool_sets.items():
        tools_to_use = [agent_tool_dict[name] for name in tool_names]
        if config.get("use_skill_lib", False) and role in ["planner", "verifier"]:
            print('using skill lib')
            tools_to_use.append(agent_tool_dict["QUERYCONFLICTS"])
        print('not using skill lib')
        # TODO
        # add the prompt for each role, now it is the same for all
        agent = agents.create_openai_tools_agent(chat, tools_to_use, prompt)
        agents_dict[role] = agents.AgentExecutor(
            agent=agent, tools=tools_to_use, verbose=True
        )

    return MultiAgent(agents_dict, agents_prompts, icao_seperation_guidelines)

def setup_agent(config, grop_api_key):
    if "multi" in config["type"]:
        return setup_multi_agent(config, grop_api_key)
    elif "single" in config["type"]:
        return setup_single_agent(config, grop_api_key)
    else:
        raise ValueError(f"Invalid agent type: {config['type']}")


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


client = initialize_simulator()
scn_files = list_scn_files(base_path, target_path)
config = load_config("config/config1.yml")
central_csv_path = "results/csv/all_results.csv"  # Central file for all results

# retrieve groq api keys
groq_api_keys = [os.getenv("GROQ_API_KEY"), os.getenv("GROQ_API_KEY_2")]
key_index = 0  # Start with the first API key

user_input = "Solve Air Traffic Conflicts"
# load agent
# agent = initialize_agent(config_path)

agent_configs = load_agent_configs(config)

agent_configs = [
    {
        "type": "single_agent",
        "model_name": "mixtral-8x7b-32768",
        "temperature": 0.9,
        "use_skill_lib": False,
    }
]
scn_files = ["TEST/ac_4/t_formation_1.scn"]

for agent_config in agent_configs:
    agent_type = agent_config["type"]
    model_name = agent_config["model_name"].replace(" ", "_").replace("-", "_")
    temperature = str(agent_config["temperature"]).replace(".", "p")

    # Path for the agent-specific CSV file
    agent_csv_path = f"results/csv/{agent_type}/{model_name}_{temperature}/results.csv"

    for scn in scn_files:
        scenario_name = "_".join(scn.split("/")[-2:]).replace(".scn", "")
        recording_directory = (
            f"results/recordings/{agent_type}/{model_name}_{temperature}"
        )
        recording_file_name = f"{scenario_name}.avi"

        num_ac = get_num_ac_from_scenario(scn)
        load_and_run_scenario(client, scn)
        conflict_type = extract_conflict_type(scn)

        recorder = ScreenRecorder(
            output_directory=recording_directory, file_name=recording_file_name
        )
        recorder.start_recording()

        success = False
        for attempt in range(5):  # Retry up to 5 times
            try:
                with CaptureAndPrintConsoleOutput() as output:
                    # Use the current API key to setup the agent
                    agent_executor = setup_agent(agent_config, groq_api_keys[key_index])
                    start_time = (
                        time.perf_counter()
                    )  # Record the start time with higher precision
                    result = agent_executor.invoke({"input": user_input})
                    elapsed_time = (
                        time.perf_counter() - start_time
                    )  # Calculate the elapsed time with higher precision
                success = True
                break  # Exit the retry loop if successful
            except Exception as e:
                print(f"Attempt {attempt + 1} failed, error: {e}")
                if attempt < 4:  # Only sleep and swap keys if it's not the last attempt
                    time.sleep(65)
                    # Swap to the next API key
                    key_index = 1 - key_index  # Toggle between 0 and 1

        if not success:
            print(f"Skipping scenario {scenario_name} after 5 failed attempts.")
            console_output = 'skipped'  # Skip to the next scenario
            elapsed_time = -1  # Mark the scenario as skipped
        else:
            console_output = output.getvalue()
            console_output = remove_ansi_escape_sequences(console_output)
        recorder.stop_recording()

        final_score, final_details = final_check()
        num_send_commands = console_output.count("Invoking: `SENDCOMMAND`")
        num_total_commands = console_output.count("Invoking:")

        print(num_send_commands, num_total_commands)
        print(elapsed_time)

        # Prepare the results dictionary
        result_data = {
            "scenario": scenario_name,
            "num_aircraft": num_ac,
            "conflict_type": conflict_type,
            "agent_type": agent_type,
            "model_name": model_name,
            "temperature": temperature,
            "runtime": elapsed_time,
            "num_total_commands": num_total_commands,
            "num_send_commands": num_send_commands,
            "score": final_score,
            "log": console_output,
            "final_details": final_details,
        }

        # Save results to both the central and agent-specific CSV files
        save_results_to_csv([result_data], central_csv_path)
        save_results_to_csv([result_data], agent_csv_path)
