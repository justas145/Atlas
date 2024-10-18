import logging
import time
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
from tools.agent_tools import GetConflictInfo
import os

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
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the bluesky/output directory
    path_to_watch = os.path.abspath(os.path.join(current_dir, '..', '..', 'bluesky', 'output'))
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


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_messages = []

    def emit(self, record):
        self.log_messages.append(self.format(record))


def monitor_too_many_requests(client):
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


def final_check(crash_log_path=None):
    if crash_log_path is None:
        crash_log_path = os.path.join(os.path.dirname(__file__), "..", "..", "bluesky", "output", "crash_log.txt")
    crash_log_path = os.path.abspath(crash_log_path)
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
    if "No conflicts detected." not in conflict_info.strip():
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


def check_preference_execution(preference, tlos_threshold, command_logs, tlos_logs):
    if preference is None:
        return None
    elif preference == "HDG":
        return 1 if all("HDG" in command for command in command_logs) else 0
    elif preference == "ALT":
        return 1 if all("ALT" in command for command in command_logs) else 0
    elif preference == "tLOS":
        if tlos_threshold is None:
            raise ValueError(
                "tLOS threshold must be specified when preference is 'tLOS'"
            )
        return 1 if all(log["tLOS"] <= tlos_threshold for log in tlos_logs) else 0
    else:
        raise ValueError(f"Invalid preference: {preference}")
