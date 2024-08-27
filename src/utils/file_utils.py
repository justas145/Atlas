import os


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
                relative_path = os.path.normpath(relative_path).replace(os.sep, '/')
                scn_files.append(relative_path)
    return scn_files


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


import os


def clear_crash_log(crash_log_filename="crash_log.txt"):
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate up to the bluesky directory
    bluesky_dir = os.path.abspath(
        os.path.join(current_dir, "..", "..", "bluesky")
    )
    # Construct the path to the crash log file
    crash_log_path = os.path.join(bluesky_dir, "output", crash_log_filename)
    # Clear the crash log file or create it if it doesn't exist
    with open(crash_log_path, "w") as file:
        file.truncate()
