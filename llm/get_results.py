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
from filelock import Timeout, FileLock
import chromadb
from langchain import agents, hub
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from contextlib import contextmanager
from prompts.agent_prompts import (
    planner_prompt,
    executor_prompt,
    verifier_prompt,
    extraction_prompt
)

dotenv.load_dotenv("../.env")
# Usage
base_path = "../bluesky/scenario"  # Adjust this to your base directory
target_path = (
    "../bluesky/scenario/TEST/Big"  # Adjust this to the directory you want to search
)

with open("prompts/ICAO_seperation_guidelines.txt", "r") as f:
    icao_seperation_guidelines = f.read()


from PIL import ImageGrab
# import pygetwindow as gw

import yaml

import cv2
import numpy as np
import os
import threading
import time
from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field
import logging
import threading
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
# Setup basic logging
# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
# )
class CrashFileHandler(PatternMatchingEventHandler):
    def __init__(self, callback):
        super().__init__(patterns=["crash_log.txt"])  # Only monitor the crash log file
        self.callback = callback
        self.last_position = 0  # Track the last position read in the file

    def on_modified(self, event):
        logging.info(f"Modification detected in: {event.src_path}")
        try:
            with open(event.src_path, "r") as file:
                file.seek(self.last_position)  # Move to the last read position
                lines = file.readlines()
                if lines:
                    logging.info(f"Detected {len(lines)} new lines in the crash log.")
                    for line in lines:
                        self.callback(line.strip())
                    self.last_position = (
                        file.tell()
                    )  # Update the last position after reading
        except Exception as e:
            logging.error(f"Error reading file {event.src_path}: {e}")


def monitor_crashes(callback):
    path_to_watch = "../bluesky/output"  # Make sure this is correct
    event_handler = CrashFileHandler(callback=callback)
    observer = Observer()
    observer.schedule(event_handler, path=path_to_watch, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def print_output(message):
    print("Crash Alert:", message)


class PlanExists(BaseModel):
    """Information about an aircraft conflict resolution plan."""

    # This doc-string is sent to the LLM as the description of the schema Metadata,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    plan_exists: Optional[bool] = Field(
        default=None,
        description="Whether an aircraft conflict resolution plan exists. Plan typically includes aircraft call signs and instructions for that aircraft. If not provided, set to False. If provided, set to True. If the plan is provided and there is a comment that there are no conflicts or any other comments, set to True.",
    )


extraction_plan_llm = ChatOpenAI(temperature=0.0, model_name="gpt-4o-mini")
extraction_plan_runnable = (
    extraction_prompt | extraction_plan_llm.with_structured_output(schema=PlanExists)
)


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

        plan_exists = extraction_plan_runnable.invoke({"text": plan}).plan_exists

        if not plan_exists:
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
            
            plan_exists = extraction_plan_runnable.invoke({"text": verifier_output}).plan_exists
            if not plan_exists:
                return {"output": verifier_output, "status": "resolved"}
            else:
                plan = verifier_output


def initialize_simulator():
    client = Client()
    client.connect("127.0.0.1", 11000, 11001)  # Adjust IP and ports as necessary
    client.update()
    initialize_client(client)

    return client

def initialize_experience_library():
# Vector DB
    selected_collection = "experience_library_v2"
    base_path = os.path.dirname(__file__)
    vectordb_path = os.path.join(base_path, "skills-library", "vectordb")
    chroma_client = chromadb.PersistentClient(path=vectordb_path)
    openai_ef = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-large",
    )
    collection = chroma_client.get_or_create_collection(
        name=selected_collection,
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"},
    )
    # Initialize the collection in agent_tools
    initialize_collection(collection)

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
        time.sleep(5)  # Wait for the scenario to load
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
        # Counting occurrences of '>CRE' and '>CRECONFS'
        count_CRE = content.count(">CRE")  # This counts all instances starting with '>CRE', including '>CRECONFS'
        count_CRECONFS = content.count(">CRECONFS")  # Specifically counts '>CRECONFS'

        # Since '>CRECONFS' is also included in '>CRE' counts, adjust the count for '>CRE'
        count_CRE_only = count_CRE - count_CRECONFS

        # Combined count
        total_aircraft_num = count_CRE_only + count_CRECONFS
        return total_aircraft_num
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

    # Create a lock file path
    lock_file = f"{output_file}.lock"
    lock = FileLock(lock_file)

    with lock:  # Acquire the lock
        # Check if the file exists to determine if headers are needed
        file_exists = os.path.isfile(output_file)

        with open(output_file, "a", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "scenario",
                "num_aircraft",
                "conflict_type",
                "conflict_with_dH",
                "agent_type",
                "model_name",
                "temperature",
                "runtime",
                "num_total_commands",
                "num_send_commands",
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
    if config.get("use_skill_lib", False):
        print('loading system prompt with exp lib')
        with open("prompts/system_with_exp_lib.txt", "r") as f:
            prompt.messages[0].prompt.template = f.read()
    else:
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
            tool
            for name, tool in agent_tool_dict.items()
            if name != "SEARCHEXPERIENCELIBRARY"
        ]
        print('not using skill lib')
    tools_to_use = list(tools_to_use)
    # Create and return the agent
    agent = agents.create_openai_tools_agent(chat, tools_to_use, prompt)
    agent_executor = agents.AgentExecutor(agent=agent, tools=tools_to_use, verbose=True, return_intermediate_steps=True)
    return agent_executor


def setup_multi_agent(config, groq_api_keys_lst):
    print("setting up multi agent with config:", config)
    agents_dict = {}
    tool_sets = {
        "planner": ["GETALLAIRCRAFTINFO", "CONTINUEMONITORING"],
        "controller": ["SENDCOMMAND"],
        "verifier": ["GETALLAIRCRAFTINFO", "CONTINUEMONITORING"],
    }

    agents_prompts = {
        "planner": planner_prompt,
        "executor": executor_prompt,
        "verifier": verifier_prompt,
    }

    for role, tool_names in tool_sets.items():
        prompt_init = hub.pull("hwchase17/openai-tools-agent")
        ### LOADING PROMPTS ###
        if role == "planner":
            if config.get("use_skill_lib", False):
                print("loading system prompt with exp lib")
                with open("prompts/planner_system_with_exp_lib.txt", "r") as f:
                    planner_system_prompt = prompt_init
                    planner_system_prompt.messages[0].prompt.template = f.read()
            else:
                with open("prompts/planner_system.txt", "r") as f:
                    planner_system_prompt = prompt_init
                    planner_system_prompt.messages[0].prompt.template = f.read()
        if role == "executor":
            with open("prompts/executor_system.txt", "r") as f:
                executor_system_prompt = prompt_init
                executor_system_prompt.messages[0].prompt.template = f.read()
        if role == "verifier":
            if config.get("use_skill_lib", False):
                print("loading system prompt with exp lib")
                with open("prompts/verifier_system_with_exp_lib.txt", "r") as f:
                    verifier_system_prompt = prompt_init
                    verifier_system_prompt.messages[0].prompt.template = f.read()
            else:
                with open("prompts/verifier_system.txt", "r") as f:
                    verifier_system_prompt = prompt_init
                    verifier_system_prompt.messages[0].prompt.template = f.read()

        ### LOADING PROMPTS ###

        groq_api_key = next(groq_key_generator)
        chat = setup_chat_model(config, groq_api_key)
        tools_to_use = [agent_tool_dict[name] for name in tool_names]
        if config.get("use_skill_lib", False) and role in ["planner", "verifier"]:
            print("using skill lib")
            tools_to_use.append(agent_tool_dict["SEARCHEXPERIENCELIBRARY"])
        print("not using skill lib")

        if role == "planner":
            prompt = planner_system_prompt
        elif role == "executor":
            prompt = executor_system_prompt
        elif role == "verifier":
            prompt = verifier_system_prompt
        agent = agents.create_openai_tools_agent(chat, tools_to_use, prompt)
        agents_dict[role] = agents.AgentExecutor(
            agent=agent, tools=tools_to_use, verbose=True, return_intermediate_steps=True
        )

    return MultiAgent(agents_dict, agents_prompts, icao_seperation_guidelines)


def setup_agent(config, grop_api_key_lst):
    if "multi" in config["type"]:
        return setup_multi_agent(config, grop_api_key_lst)
    elif "single" in config["type"]:
        groq_api_key = next(groq_key_generator)
        return setup_single_agent(config, groq_api_key)
    else:
        raise ValueError(f"Invalid agent type: {config['type']}")


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_messages = []

    def emit(self, record):
        self.log_messages.append(self.format(record))


def monitor_too_many_requests():
    # Configure a logger for capturing specific log messages
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    list_handler = ListHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    list_handler.setFormatter(formatter)
    logger.addHandler(list_handler)

    while True:
        # Simulating continuous log capturing
        time.sleep(0.5)  # Check every 0.5 seconds for new logs
        for log in list_handler.log_messages:
            if "429 Too Many Requests" in log:
                # print_output("Too many requests, pausing operations...")
                client.send_event(b"STACK", "HOLD")
                list_handler.log_messages.clear()  # Clear the log messages after handling
                break


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


initialize_experience_library()
client = initialize_simulator()
scn_files = list_scn_files(base_path, target_path)

print(scn_files)
config = load_config("config/config1.yml")
central_csv_path = "results/csv/all_results.csv"  # Central file for all results

# retrieve groq api keys
groq_api_keys_lst = load_groq_api_keys()
groq_key_generator = cyclic_key_generator(groq_api_keys_lst)

user_input = "You are an air traffic controller with tools. Solve aircraft conflict. Solve until there are no more conflicts."
# load agent
# agent = initialize_agent(config_path)

agent_configs = load_agent_configs(config)

record_screen = False

agent_configs = [
    {
        "type": "multi_agent",
        "model_name": "llama3-70b-8192",
        "temperature": 0.3,
        "use_skill_lib": True,
    },
    {
        "type": "multi_agent",
        "model_name": "gpt-4o-2024-08-06",
        "temperature": 0.3,
        "use_skill_lib": True,
    },
    {
        "type": "single_agent",
        "model_name": "llama3-70b-8192",
        "temperature": 0.3,
        "use_skill_lib": True,
    },
    {
        "type": "single_agent",
        "model_name": "gpt-4o-2024-08-06",
        "temperature": 0.3,
        "use_skill_lib": True,
    },
]

if __name__ == "__main__":
    # Run the crash monitoring in a background thread
    threading.Thread(target=monitor_crashes, args=(print_output,), daemon=True).start()
    # monitor too many request -> pause simulator in the background
    threading.Thread(target=monitor_too_many_requests, daemon=True).start()
    ##############################

    for scn in scn_files:
        for agent_config in agent_configs:

            # Initialize variables for rerun logic
            rerun_scenario = True
            attempts_count = 0
            max_attempts = 5
            while rerun_scenario and attempts_count < max_attempts:
                attempts_count += 1

                agent_type = agent_config["type"]
                model_name = agent_config["model_name"].replace(" ", "_").replace("-", "_")
                temperature = str(agent_config["temperature"]).replace(".", "p")

                # Path for the agent-specific CSV file
                agent_csv_path = f"results/csv/{agent_type}/{model_name}_{temperature}/results.csv"
                if 'no_dH' in scn:
                    conflict_with_dH = False
                elif 'dH' in scn:
                    conflict_with_dH = True
                else:
                    conflict_with_dH = None
                scenario_name = "_".join(scn.split("/")[-3:]).replace(".scn", "")

                num_ac = get_num_ac_from_scenario(scn)
                load_and_run_scenario(client, scn)
                conflict_type = extract_conflict_type(scn) 
                if record_screen:
                    recording_directory = (
                        f"results/recordings/{agent_type}/{model_name}_{temperature}"
                    )
                    recording_file_name = f"{scenario_name}.avi"
                    recorder = ScreenRecorder(
                        output_directory=recording_directory, file_name=recording_file_name
                    )
                    recorder.start_recording()

                success = False
                for attempt in range(5):  # Retry up to 5 times
                    try:
                        with CaptureAndPrintConsoleOutput() as output:
                            # Use the current API key to setup the agent
                            agent_executor = setup_agent(agent_config, groq_api_keys_lst)
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
                            time.sleep(5)
                            # Swap to the next API key

                if not success:
                    print(f"Skipping scenario {scenario_name} after 5 failed attempts.")
                    console_output = 'skipped'  # Skip to the next scenario
                    elapsed_time = -1  # Mark the scenario as skipped
                else:
                    console_output = output.getvalue()
                    console_output = remove_ansi_escape_sequences(console_output)

                if record_screen:
                    recorder.stop_recording()
                    
                rerun_scenario = 'tool-use>' in console_output
                if rerun_scenario:
                    print('reruning the scenario, because of TOOL ERROR')

            final_score, final_details = final_check()
            print('FINAL SCORE:' , final_score)
            num_send_commands = console_output.count("Invoking: `SENDCOMMAND`")
            num_total_commands = console_output.count("Invoking:")

            print(num_send_commands, num_total_commands)
            print(elapsed_time)

            # Prepare the results dictionary
            result_data = {
                "scenario": scenario_name,
                "num_aircraft": num_ac,
                "conflict_type": conflict_type,
                "conflict_with_dH": conflict_with_dH,
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
