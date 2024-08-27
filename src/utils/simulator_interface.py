import sys
import time
from contextlib import contextmanager
from io import StringIO
import os
from bluesky.network.client import Client
from .file_utils import clear_crash_log

def initialize_simulator():
    client = Client()
    client.connect("127.0.0.1", 11000, 11001)  # Adjust IP and ports as necessary
    client.update()
    return client


def initialize_client():
    client.connect("127.0.0.1", 11000, 11001)
    client.update()
    print("Connected to BlueSky")


@contextmanager
def capture_stdout():
    new_stdout = StringIO()
    old_stdout = sys.stdout
    sys.stdout = new_stdout
    try:
        yield new_stdout
    finally:
        sys.stdout = old_stdout


def receive_bluesky_output(client):
    complete_output = ""
    empty_output_count = 0

    while True:
        with capture_stdout() as captured:
            client.update()
        new_output = captured.getvalue()

        if not new_output.strip():
            empty_output_count += 1
        else:
            empty_output_count = 0
            complete_output += new_output

        if empty_output_count >= 5:
            break

    client.update()
    return complete_output


def load_and_run_scenario(client, scenario_path):
    try:
        clear_crash_log()
        # client.send_event(b"STACK", f"IC {scenario_path}")
        client.send_event(b"STACK", f"IC {scenario_path}")

        client.update()
        print(f"Loaded scenario: {scenario_path}")
        time.sleep(5)  # Wait for the scenario to load
        # clear the output buffer
        out = receive_bluesky_output(client)

    except Exception as e:
        print(f"Failed to load scenario {scenario_path}: {str(e)}")
