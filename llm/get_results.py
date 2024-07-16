import os
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
from contextlib import contextmanager

dotenv.load_dotenv("../.env")
# Usage
base_path = "../bluesky/scenario"  # Adjust this to your base directory
target_path = (
    "../bluesky/scenario/TEST"  # Adjust this to the directory you want to search
)


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
        time.sleep(1)  # Wait for the scenario to load
        # clear the output buffer
        out = receive_bluesky_output()
        print(out)

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


def final_check(crash_log_path="../bluesky/output/crash_log.txt"):
    """
    Performs the final checks after the agent has completed its task.

    Args:
    crash_log_path (str): Path to the crash log file.
    get_conflict_info (func): Function that checks for conflicts and returns conflict details.

    Returns:
    tuple: (score, details) where score is the evaluation score and details contain either crash or conflict information.
    """
    # Check for a crash
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
    conflict_info = GetConflictInfo("SHOWTCPA")
    if (
        conflict_info.strip() != "No conflicts detected."
        
    ):  # Adjusting condition to explicitly handle the no-conflict scenario
        print(conflict_info)
        return 0, conflict_info  # Conflicts detected, score 0, return conflict details
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
            "agent_type",
            "model_name",
            "temperature",
            "score",
            "log",
            "final_details",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header only if the file did not exist
        if not file_exists:
            writer.writeheader()

        # Write results
        for result in results:
            writer.writerow(result)


def setup_agent(config):
    # Load system prompts
    prompt = hub.pull("hwchase17/openai-tools-agent")
    with open("prompts/system.txt", "r") as f:
        prompt.messages[0].prompt.template = f.read()
    print('setting up agent with config:', config)
    # Initialize LLM model
    chat = ChatGroq(temperature=config["temperature"], model_name=config["model_name"])
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

client = initialize_simulator()
scn_files = list_scn_files(base_path, target_path)
config = load_config("config/config1.yml")
central_csv_path = "results/csv/all_results.csv"  # Central file for all results

user_input = "Solve Air Traffic Conflicts"
# load agent
# agent = initialize_agent(config_path)

agent_config = config['agents'][0]
print(agent_config)

for agent_config in config['agents']:
    agent_type = agent_config['type']
    model_name = agent_config['model_name'].replace(' ', '_').replace('-', '_')
    temperature = str(agent_config['temperature']).replace('.', 'p')

    agent_executor = setup_agent(agent_config)

    for scn in scn_files[0:1]:

        scenario_name = scn.split("/")[-1].replace(
            ".scn", ""
        )  # Adjust path handling as needed
        recording_directory = f"results/recordings/{agent_type}/{model_name}_{temperature}"
        recording_file_name = f"{scenario_name}.avi"

        num_ac = get_num_ac_from_scenario(scn)
        print(num_ac)
        load_and_run_scenario(client, scn)
        print(scn)
        conflict_type = extract_conflict_type(scn)
        print(conflict_type)
        # # run agent based on config
        recorder = ScreenRecorder(
        output_directory=recording_directory, file_name=recording_file_name)
        recorder.start_recording()
        with CaptureAndPrintConsoleOutput() as output:
            result = agent_executor.invoke({"input": user_input})
        #time.sleep(10)
        recorder.stop_recording()
        console_output = output.getvalue()
        # # final conflict check
        final_score, final_details = final_check()
        print(final_score)
        print(final_details)
        # print(score)
        # print(output)
        # # get results

        # # append results to results.csv
