import os
import sys
sys.path.append("../bluesky")
from bluesky.network.client import Client
import dotenv

dotenv.load_dotenv("../.env")
# Usage
base_path = "../bluesky/scenario"  # Adjust this to your base directory
target_path = (
    "../bluesky/scenario/tt"  # Adjust this to the directory you want to search
)


def initialize_simulator():
    client = Client()
    client.connect("127.0.0.1", 11000, 11001)  # Adjust IP and ports as necessary
    client.update()

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
                scn_files.append(relative_path)
    return scn_files


def load_and_run_scenario(client, scenario_path):
    scenario_path = scenario_path.replace("\\", "/")
    try:
        # client.send_event(b"STACK", f"IC {scenario_path}")
        client.send_event(b"STACK", f"IC TEST/2_ac/converging_1.scn")

        client.update()
        print(f"Loaded scenario: {scenario_path}")

    except Exception as e:
        print(f"Failed to load scenario {scenario_path}: {str(e)}")


# scn_files = list_scn_files(base_path, target_path)
client = initialize_simulator()
# load_and_run_scenario(client, scn_files[0])

# print(scn_files[0])
client.send_event(b"STACK", f"IC simple_test/ac_2/converging_1.scn")
