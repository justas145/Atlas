import time
from pathlib import Path

base_dir = Path("../bluesky/output")
base_dir.mkdir(parents=True, exist_ok=True)
# Define the full path to the crash log
log_file_path = base_dir / "crash_log.txt"
def monitor_crash_log():
    last_position = 0
    while True:
        try:
            with open(log_file_path, "r") as file:
                file.seek(last_position)
                lines = file.readlines()
                if lines:
                    print("New crash data:", lines)
                    last_position = file.tell()
            time.sleep(1)  # Adjust frequency as needed
        except FileNotFoundError:
            print("File not found. Waiting for new crash log.")
            pass


if __name__ == "__main__":
    monitor_crash_log()
