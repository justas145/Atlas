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
    #GetConflictInfo,
    receive_bluesky_output,
    agent_tool_dict,
)
import time
from filelock import Timeout, FileLock
import chromadb
from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain import agents, hub
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from contextlib import contextmanager
from prompts.agent_prompts import (
    planner_prompt,
    executor_prompt,
    verifier_prompt,
    extraction_prompt,
)

dotenv.load_dotenv("../.env")
# Usage
base_path = "../bluesky/scenario"  # Adjust this to your base directory
target_path = (
    "../bluesky/scenario/TEST/Big"  # Adjust this to the directory you want to search
)

with open("prompts/ICAO_seperation_guidelines.txt", "r") as f:
    icao_seperation_guidelines = f.read()


# from PIL import ImageGrab
# import pygetwindow as gw

import yaml

import cv2
import numpy as np
import os
import threading
import time

import logging
import threading
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler


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
        

        if "no conflicts" in plan.lower() and len(plan) < 100:
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


def initialize_experience_library():
    # Vector DB
    selected_collection = "experience_library_v3"
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


def load_and_run_scenario(client, scenario_path):
    try:
        # client.send_event(b"STACK", f"IC {scenario_path}")
        client.send_event(b"STACK", f"IC {scenario_path}")

        client.update()
        print(f"Loaded scenario: {scenario_path}")
        time.sleep(5)  # Wait for the scenario to load
        # clear the output buffer
        out = receive_bluesky_output()

    except Exception as e:
        print(f"Failed to load scenario {scenario_path}: {str(e)}")


def setup_chat_model(config, groq_api_key):
    if "gpt" in config["model_name"]:
        chat = ChatOpenAI(
            temperature=config["temperature"], model_name=config["model_name"]
        )
    else:
        chat = ChatGroq(
            temperature=config["temperature"],
            model_name=config["model_name"],
            api_key=groq_api_key,
        )
    return chat


def setup_single_agent(config, groq_api_key):
    # Load system prompts
    prompt = hub.pull("hwchase17/openai-tools-agent")
    if config.get("use_skill_lib", False):
        print("loading system prompt with exp lib")
        with open("prompts/system_with_exp_lib.txt", "r") as f:
            prompt.messages[0].prompt.template = f.read()
    else:
        with open("prompts/system.txt", "r") as f:
            prompt.messages[0].prompt.template = f.read()
    print("setting up agent with config:", config)
    # Initialize LLM model
    chat = setup_chat_model(config, groq_api_key)
    # Decide on tools to include based on the 'use_skill_lib' configuration
    if agent_config.get("use_skill_lib", False):
        tools_to_use = (
            agent_tool_dict.values()
        )  # Use all available tools if skill lib is to be used
        print("using skill lib")
    else:
        tools_to_use = [
            tool
            for name, tool in agent_tool_dict.items()
            if name != "SEARCHEXPERIENCELIBRARY"
        ]
        print("not using skill lib")
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
            agent=agent, tools=tools_to_use, verbose=True
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
central_csv_path = "results/csv/all_results.csv"  # Central file for all results

# retrieve groq api keys
groq_api_keys_lst = load_groq_api_keys()
groq_key_generator = cyclic_key_generator(groq_api_keys_lst)

user_input = "You are an air traffic controller with tools. Solve aircraft conflict. Solve until there are no more conflicts."
# load agent
# agent = initialize_agent(config_path)


record_screen = False
# llama3-70b-8192
# llama-3.1-70b-versatile
# gpt-4o-2024-08-06
agent_configs = [
    {
        "type": "single_agent",
        "model_name": "llama3-70b-8192",
        "temperature": 0.3,
        "use_skill_lib": True,
    },
]

if __name__ == "__main__":
    threading.Thread(target=monitor_too_many_requests, daemon=True).start()

    # Run the crash monitoring in a background thread
    # ac_4_no_dH_parallel_5
    # ac_2_dH_head-on_8
    # ac_4_no_dH_t-formation_1
    # ac_4_no_dH_t-formation_5
    # ac_3_dH_parallel_6
    # ac_2_dH_head-on_7
    # ac_3_dH_head-on_10
    # ac_3_dH_t-formation_8
    scn_files = ["TEST/Big/ac_3/dH/t-formation_8.scn"]
    for scn in scn_files:
        for agent_config in agent_configs:

            rerun_scenario = True
            attempts_count = 0
            max_attempts = 5

            while rerun_scenario and attempts_count < max_attempts:
                attempts_count += 1

                agent_type = agent_config["type"]
                model_name = agent_config["model_name"].replace(" ", "_").replace("-", "_")

                scenario_name = "_".join(scn.split("/")[-3:]).replace(".scn", "")

                load_and_run_scenario(client, scn)

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
                        if (
                            attempt < 4
                        ):  # Only sleep and swap keys if it's not the last attempt
                            time.sleep(5)
                            # Swap to the next API key

                if not success:
                    print(f"Skipping scenario {scenario_name} after 5 failed attempts.")
                    console_output = "skipped"  # Skip to the next scenario
                    elapsed_time = -1  # Mark the scenario as skipped
                else:
                    console_output = output.getvalue()

                rerun_scenario = 'tool-use>' in console_output  # Check if rerun is needed
                if rerun_scenario:
                    print(f"Rerunning scenario {scenario_name} due to detection of 'tool-use>' in output.")

print(result)
