import re
import os
import csv
from filelock import FileLock

def parse_conflict_data(conflict_output):
    """Parse the conflict output into a structured dictionary and capture the number of aircraft pairs in conflict."""
    if "No conflicts detected." in conflict_output:
        return {}, {}, 0  # Include zero as the number of aircraft in conflicts

    conflict_info = {}
    altitude_info = {}
    conflict_pattern = r"(\w+ - \w+) \| TCPA: ([-\d\.]+) sec \| Heading Difference: ([-\d\.]+) deg \| Distance: ([-\d\.]+) Nautical miles \| Vertical Separation: ([-\d\.]+) ft \| Horizontal Distance: ([-\d\.]+) Nautical miles \| DCPA: ([-\d\.]+) Nautical miles \| tLOS: ([-\d\.]+) sec"

    altitude_pattern = (
        r"Aircraft (\w+): Altitude ([\d\.]+) ft \((ascending|descending|level)\)"
    )
    num_conflicts_pattern = r"Number of aircraft pairs in conflict: (\d+)"

    # Parsing conflict data
    conflict_matches = re.findall(conflict_pattern, conflict_output)
    for match in conflict_matches:
        conflict_info[match[0]] = {
            "TCPA": float(match[1]),
            "Heading Difference": float(match[2]),
            "Distance": float(match[3]),
            "Vertical Separation": float(match[4]),
            "Horizontal Distance": float(match[5]),
            "DCPA": float(match[6]),
            "tLOS": float(match[7]),
        }

    # Parsing altitude data
    altitude_matches = re.findall(altitude_pattern, conflict_output)
    for match in altitude_matches:
        altitude_info[match[0]] = {"Altitude": float(match[1]), "Status": match[2]}

    # Parsing number of aircraft in conflict
    num_conflicts_match = re.search(num_conflicts_pattern, conflict_output)
    num_conflicts = int(num_conflicts_match.group(1)) if num_conflicts_match else 0

    return conflict_info, altitude_info, num_conflicts


def generate_compact_conflict_report(
    initial_data, updated_data, initial_altitudes, updated_altitudes, num_conflicts
):
    """Generates a report showing changes in conflict data and subsequent altitude changes."""
    report = ["Aircraft Pairs in Conflict and their TCPA (sec):"]
    units = {
        "TCPA": "sec",
        "Heading Difference": "deg",
        "Distance": "Nautical miles",
        "Vertical Separation": "ft",
        "Horizontal Distance": "Nautical miles",
        "DCPA": "Nautical miles",
        "tLOS": "sec",
    }

    all_keys = set(initial_data.keys()).union(updated_data.keys())
    for key in all_keys:
        if key in initial_data and key in updated_data:
            # Display all conflict attributes with units, showing changes
            conflict_change_text = " | ".join(
                f"{field}: {initial_data[key][field]} {units[field]} -> {updated_data[key][field]} {units[field]}"
                for field in initial_data[key]
            )
            report.append(f"{key} | {conflict_change_text} \n")
        elif key in updated_data:
            # For new conflicts, display initial and final states if available, otherwise just final
            new_conflict_values = " | ".join(
                f"{field}: {updated_data[key][field]} {units[field]}"
                for field in updated_data[key]
            )
            report.append(f"{key} | New conflict detected | {new_conflict_values} \n")

    if num_conflicts > 0:  # Include number of aircraft in conflicts if relevant
        report.append(f"Number of aircraft pairs in conflict: {num_conflicts}\n")

    # Append altitude information for all aircraft if available
    if initial_altitudes and updated_altitudes:  # Check if any altitude data exists
        altitude_info = ["Aircraft Altitude Information:"]
        for (
            aircraft_id
        ) in (
            initial_altitudes.keys()
        ):  # Assume all aircraft from initial are present in updated
            altitude_info.append(
                f"{aircraft_id}: Altitude {initial_altitudes[aircraft_id]['Altitude']} ft -> {updated_altitudes[aircraft_id]['Altitude']} ft ({updated_altitudes[aircraft_id]['Status']})"
            )
        report.extend(altitude_info)  # Append altitude info to the main report
    elif updated_altitudes:
        report.append("Aircraft Altitude Information:")
        for aircraft_id in updated_altitudes.keys():
            report.append(
                f"{aircraft_id}: Altitude {updated_altitudes[aircraft_id]['Altitude']} ft ({updated_altitudes[aircraft_id]['Status']})"
            )

    if len(report) == 1:
        return "No conflicts detected."

    return "\n".join(report)


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
                "json_results",
                "experience_library",
                "preference",
                "preference_executed",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header only if the file did not exist
            if not file_exists:
                writer.writeheader()

            # Write results
            for result in results:
                # Create a new dictionary, replacing None with empty string
                filtered_result = {k: ('' if v is None else v) for k, v in result.items() if k in fieldnames}
                writer.writerow(filtered_result)


def get_num_ac_from_scenario(scenario_path):
    # Construct the full path to the scenario file

    try:
        # Open the scenario file and read its content
        with open(scenario_path, "r") as file:
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
        print(f"File not found: {scenario_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


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
