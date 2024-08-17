import sys
import time
from contextlib import contextmanager
from io import StringIO
from langchain_core.tools import tool as langchain_tool

client = None  # Module-level variable for the client
collection = None  # Module-level variable for the collection


def initialize_client(new_client):
    global client
    client = new_client


def initialize_collection(new_collection):
    global collection
    collection = new_collection


@contextmanager
def capture_stdout():
    new_stdout = StringIO()
    old_stdout = sys.stdout
    sys.stdout = new_stdout
    try:
        yield new_stdout
    finally:
        sys.stdout = old_stdout


def receive_bluesky_output():
    complete_output = ""
    empty_output_count = 0  # Track consecutive empty outputs

    while True:
        with capture_stdout() as captured:
            client.update()
        new_output = captured.getvalue()

        # Check if the current output is empty
        if not new_output.strip():
            empty_output_count += 1  # Increment counter for empty outputs
        else:
            empty_output_count = 0  # Reset counter if output is not empty
            complete_output += new_output  # Add non-empty output to complete output

        # If there are two consecutive empty outputs, break the loop
        if empty_output_count >= 5:
            break

    # It's assumed you want to keep the last update outside the loop
    client.update()

    return complete_output


@langchain_tool("GETALLAIRCRAFTINFO")
def GetAllAircraftInfo(command: str = "GETACIDS"):
    """
    Get each aircraft information at current time: Position [Pos] (lat, lon), Heading [Hdg] (deg),
    Track [Trk](deg), Altitude [Alt](ft), Vertical speed [V/S](feet per mintute. Negative V/S - flying down, Positive V/S - flying up, 0 - stays at same altitude), Speed [CAS/TAS/GS] (calibrated/true/ground air speed, knots per second) and mach number.

    Parameters:
    - command: str (default 'GETACIDS')

    Example usage:
    - GetAllAircraftInfo('GETACIDS')

    Returns:
    - str: all aircraft information
    """

    command = command.replace('"', "").replace("'", "")
    command = command.split("\n")[0]
    client.send_event(b"STACK", "OP")
    client.send_event(b"STACK", "GETACIDS")
    time.sleep(0.8)
    sim_output = receive_bluesky_output()
    return sim_output


@langchain_tool("GETCONFLICTINFO")
def GetConflictInfo(command: str = "SHOWTCPA"):
    """
    Use this tool to get information on aircraft pairs in
    conflict. It gives you Time to Closest Point of Approach (TCPA),
    Quadrantal Direction (QDR), separation distance, Closest Point of Approach
    distance (DCPA), and Time of Loss of Separation (tLOS).

    Parameters:
    - command: str (default 'SHOWTCPA')

    Example usage:
    - GetConflictInfo('SHOWTCPA')

    Returns:
    - str: conflict information between aircraft pairs
    """
    client.send_event(b"STACK", "OP")
    client.send_event(b"STACK", "SHOWTCPA")
    time.sleep(0.8)
    sim_output = receive_bluesky_output()
    return sim_output


# @langchain_tool("CONTINUEMONITORING")
# def ContinueMonitoring(duration: int = 8):
#     """Monitor for conflicts between aircraft pairs for a specified time.

#     Parameters:
#     - duration (int): The time in seconds to monitor for conflicts. Default and min is 8 seconds. Maximum duration is 16 seconds.

#     Example usage:
#     - ContinueMonitoring(8)

#     Returns:
#     - str: The conflict information at the start of duration and after duration seconds.
#     """
#     # ensure duration is an integer
#     duration = int(duration)
#     duration = 8 if duration < 8 else duration
#     duration = 16 if duration > 16 else duration
#     client.send_event(b"STACK", "OP")
#     client.send_event(b"STACK", "SHOWTCPA")
#     time.sleep(0.8)
#     conflict_info1 = receive_bluesky_output()
#     time.sleep(duration)
#     client.send_event(b"STACK", "OP")
#     client.send_event(b"STACK", "SHOWTCPA")
#     time.sleep(0.8)
#     conflict_info2 = receive_bluesky_output()

#     final_output = conflict_info1 + f"\n After {duration} seconds \n" + conflict_info2
#     return final_output


@langchain_tool("SENDCOMMAND")
def SendCommand(command: str):
    """
    Sends a command to the simulator and returns the output.
    You can send a single command at a time.
    Parameters:
    - command (str): command to send to the simulator.
    Returns:
    str: The output from the simulator.
    """

    command = command.replace('"', "").replace("'", "")
    command = command.split("\n")[0]
    
    # fixes bug in the simulator, where default vertical speed is not 1500 when aircraft already has a vertical speed
    if "ALT" in command:
        # add a vertical speed of 3000 at the end of a command to speed up the process
        command = command + " 3000"
    
    client.send_event(b"STACK", "OP")
    client.send_event(b"STACK", command)
    time.sleep(0.8)
    sim_output = receive_bluesky_output()
    if sim_output == "":
        return "Command executed successfully."
    if "Unknown command" in sim_output:
        return (
            sim_output
            + "\n"
            + "Please use a correct command."
        )
    return sim_output


@langchain_tool("SEARCHEXPERIENCELIBRARY")
def SearchExperienceLibrary(
    conflict_description: str,
    num_ac: int,
    conflict_formation: str,
):
    """
    Search in the experience library for a similar conflict and its resolution. Only use it after you aquired aircraft information and conflict details.

    :param conflict_description: Detailed description of the conflict in a couple of sentences, e.g., how each aircraft are positioned and headed relative to one another without numbers.
    :param num_ac: Number of aircraft involved in the conflict.
    :param conflict_formation: Formation of the conflict, options include "Head-On Formation", "T-Formation", "Parallel Formation", "Converging Formation".
    :return: Document with similar conflict and advices on how to resolve it or no document if nothing similar was found.
    """

    where_full = {
        "$and": [
            {"num_ac": num_ac},
            {"conflict_formation": conflict_formation},

        ]
    }

    where_partial_1 = {
        "$and": [
            {"num_ac": num_ac},
        ]
    }

    search_orders = [
        (where_full, "Full"),
        (where_partial_1, "Partial 1"),
        (None, "No filters"),
    ]

    for where, label in search_orders:
        try:
            query_results = collection.query(
                query_texts=[conflict_description], n_results=1, where=where
            )
            if query_results["documents"] and query_results["documents"][0]:
                doc = query_results["documents"][0][0]
                print(f"{label} query results:", doc)
                return doc + "\n\n" + "Remember this is only a similar conflict and not identical. Use the information wisely."
        except Exception as e:
            print(f"Error with {label} query:", e)

    return "No similar conflict found in the database."


@langchain_tool("GETBLUESKYCOMMANDS")
def GetBlueskyCommands(ids: str) -> str:
    """
    Get the commands' documentation from the BlueSky database for the given ids.

    Parameters:
    - ids: string of ids seperated by comma (the ids of the commands to retrieve)
    
    Example usage:
    - get_bluesky_commands("HDG, VS, ALT, ...")

    Returns:
    - str: the commands from the BlueSky database for the given ids
    """
    commands_lst = list(set([item.strip() for item in ids.split(",")]))
    documents_lst = collection.get(ids=commands_lst)["documents"]
    documents_str = ""
    for doc in documents_lst:
        documents_str += doc + "\n\n ############################## \n\n"
    return documents_str


@langchain_tool("QUERYDATABASE")
def QueryDatabase(input: str):
    """Query skill database

    If you want to send a command to a simulator, please first search for the appropriate command.
    For example, if you want to create an aircraft, search for 'how do I create an aircraft'.

    Parameters:
    - input: str (the query to search for)

    Returns:
    - list: the top 5 results from the database
    """

    query_results = collection.query(query_texts=[input], n_results=5)
    return query_results


import re


@langchain_tool("CONTINUEMONITORING")
def ContinueMonitoring(duration: int = 10):
    """Monitor for conflicts between aircraft pairs for a specified time.

    Parameters:
    - duration (int): The time in seconds to monitor for conflicts. Default is 10 seconds, minimum is 8 seconds, maximum duration is 16 seconds.

    Returns:
    - str: A compact representation of conflict information initially and changes after the specified duration.
    """
    # Ensure duration is within the allowed range
    duration = max(8, min(duration, 16))
    client.send_event(b"STACK", "OP")
    client.send_event(b"STACK", "SHOWTCPA")
    time.sleep(1)
    initial_conflicts = receive_bluesky_output()

    time.sleep(duration)
    client.send_event(b"STACK", "OP")
    client.send_event(b"STACK", "SHOWTCPA")
    time.sleep(1)
    updated_conflicts = receive_bluesky_output()

    # Parse the conflict data from the output
    initial_conflict_data, initial_altitude_data, initial_num_conflicts = parse_conflict_data(initial_conflicts)
    updated_conflict_data, updated_altitude_data, updated_num_conflicts = parse_conflict_data(updated_conflicts)
    

    # Generate a compact conflict report based on the parsed data
    final_output = generate_compact_conflict_report(initial_conflict_data, updated_conflict_data, initial_altitude_data, updated_altitude_data, updated_num_conflicts)
    return final_output


def parse_conflict_data(conflict_output):
    """Parse the conflict output into a structured dictionary and capture the number of aircraft in conflict."""
    if "No conflicts detected." in conflict_output:
        return {}, {}, 0  # Include zero as the number of aircraft in conflicts

    conflict_info = {}
    altitude_info = {}
    conflict_pattern = r"(\w+ - \w+) \| TCPA: ([-\d\.]+) sec \| QDR: ([-\d\.]+) deg \| Distance: ([-\d\.]+) Nautical miles \| Vertical Separation: ([-\d\.]+) ft \| Horizontal Distance: ([-\d\.]+) Nautical miles \| DCPA: ([-\d\.]+) Nautical miles \| tLOS: ([-\d\.]+) sec"

    altitude_pattern = (
        r"Aircraft (\w+): Altitude ([\d\.]+) ft \((ascending|descending|level)\)"
    )
    num_conflicts_pattern = r"Number of aircraft in conflict: (\d+)"

    # Parsing conflict data
    conflict_matches = re.findall(conflict_pattern, conflict_output)
    for match in conflict_matches:
        conflict_info[match[0]] = {
            "TCPA": float(match[1]),
            "QDR": float(match[2]),
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
        "QDR": "deg",
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
        # elif key in initial_data:
        #     report.append(f"{key} | Conflict resolved")
        elif key in updated_data:
            # For new conflicts, display initial and final states if available, otherwise just final
            new_conflict_values = " | ".join(
                f"{field}: {updated_data[key][field]} {units[field]}"
                for field in updated_data[key]
            )
            report.append(f"{key} | New conflict detected | {new_conflict_values} \n")
            
    
    if num_conflicts > 0:  # Include number of aircraft in conflicts if relevant
        report.append(f"Number of aircraft in conflict: {num_conflicts}\n")


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


# Adjust the invocation of these functions in ContinueMonitoring accordingly, ensuring you pass both conflict and altitude data correctly.


agent_tools_list = [
    GetAllAircraftInfo,
    GetConflictInfo,
    SendCommand,
    ContinueMonitoring,
    SearchExperienceLibrary,
]


agent_tool_dict = {tool.name: tool for tool in agent_tools_list}
