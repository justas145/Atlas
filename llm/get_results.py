import os
import sys
sys.path.append("../bluesky")
from bluesky.network.client import Client
import dotenv
from agent_tools import (
    initialize_client,
    initialize_collection,
    agent_tools_list,
    GetConflictInfo,
)

dotenv.load_dotenv("../.env")
# Usage
base_path = "../bluesky/scenario"  # Adjust this to your base directory
target_path = (
    "../bluesky/scenario/TEST"  # Adjust this to the directory you want to search
)


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


def load_and_run_scenario(client, scenario_path):
    try:
        # client.send_event(b"STACK", f"IC {scenario_path}")
        client.send_event(b"STACK", f"IC {scenario_path}")

        client.update()
        print(f"Loaded scenario: {scenario_path}")

    except Exception as e:
        print(f"Failed to load scenario {scenario_path}: {str(e)}")

client = initialize_simulator()
scn_files = list_scn_files(base_path, target_path)
load_and_run_scenario(client, scn_files[0])
