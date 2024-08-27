import os
import sys
import time
import threading
import cv2
import numpy as np
from PIL import ImageGrab

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
        count_CRE = content.count(
            ">CRE"
        )  # This counts all instances starting with '>CRE', including '>CRECONFS'
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
                    print(f"dcpa is {dcpa_meters}, crash detected!")
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
