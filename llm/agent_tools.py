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
    Get each aircraft information at current time: position, heading (deg),
    track (deg), altitude, V/S (vertical speed), calibrated, true and ground
    speed and mach number. Input is 'GETACIDS'.

    Parameters:
    - command: str (default 'GETACIDS')

    Example usage:
    - GetAllAircraftInfo('GETACIDS')

    Returns:
    - str: all aircraft information
    """

    command = command.replace('"', "").replace("'", "")
    command = command.split("\n")[0]
    print(f"LLM input:{command}")

    client.send_event(b"STACK", "GETACIDS")
    time.sleep(0.8)
    sim_output = receive_bluesky_output()
    return sim_output


@langchain_tool("GETCONFLICTINFO")
def GetConflictInfo(command: str = "SHOWTCPA"):
    """
    Use this tool to identify and get vital information on aircraft pairs in
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

    client.send_event(b"STACK", "SHOWTCPA")
    time.sleep(0.8)
    sim_output = receive_bluesky_output()
    return sim_output


@langchain_tool("CONTINUEMONITORING")
def ContinueMonitoring(duration: int = 5):
    """Monitor for conflicts between aircraft pairs for a specified time.

    Parameters:
    - duration (int): The time in seconds to monitor for conflicts. Default is 5 seconds. Maximum duration is 15 seconds.

    Example usage:
    - ContinueMonitoring(5)

    Returns:
    - str: The conflict information between aircraft pairs throughout the monitoring period.
    """
    # ensure duration is an integer
    duration = int(duration)
    duration = 15 if duration > 15 else duration
    client.send_event(b"STACK", "SHOWTCPA")
    time.sleep(0.8)
    conflict_info1 = receive_bluesky_output()
    client.send_event(b"STACK", "SHOWTCPA")
    time.sleep(duration)
    conflict_info2 = receive_bluesky_output()

    final_output = conflict_info1 + f"\n After {duration} seconds \n" + conflict_info2
    return final_output


@langchain_tool("SENDCOMMAND")
def SendCommand(command: str):
    """
    Sends a command with optional arguments to the simulator and returns the output.
    You should only send 1 command at a time.

    Parameters:
    - command (str): The command to send to the simulator. Can only be a single command,
      with no AND or OR operators.

    Example usage:
    - SendCommand('COMMAND_NAME ARG1 ARG2 ARG3 ...) # this command requires arguments
    - SendCommand('COMMAND_NAME') # this command does not require arguments

    Returns:
    str: The output from the simulator.
    """

    print(command)
    command = command.replace('"', "").replace("'", "")
    command = command.split("\n")[0]
    client.send_event(b"STACK", command)
    time.sleep(0.8)
    sim_output = receive_bluesky_output()
    if sim_output == "":
        return "Command executed successfully."
    if "Unknown command" in sim_output:
        return (
            sim_output
            + "\n"
            + "Please use a tool QueryDatabase to search for the correct command."
        )
    return sim_output


@langchain_tool("QUERYCONFLICTS")
def QueryConflicts(input: str, num_ac: int, conflict_formation: str):
    """
    Query for a similar conflict and its resolution in the database.
    """
    where_full = {
        "$and": [
            {"num_ac": num_ac},
            {"conflict_formation": conflict_formation},
        ]
    }
    where_partial = {"num_ac": num_ac}
    print(3)

    try:
        query_results = collection.query(
            query_texts=[input], n_results=1, where=where_full
        )
        print("Full query results:", query_results)
        doc = query_results["documents"][0][0] if query_results["documents"] else ""
        print(1)
        if doc:
            return doc
    except Exception as e:
        print("Error with full query:", e)

    try:
        query_results = collection.query(
            query_texts=[input], n_results=1, where=where_partial
        )
        print("Partial query results:", query_results)
        doc = query_results["documents"][0][0] if query_results["documents"] else ""
        print(2)
        if doc:
            return doc
    except Exception as e:
        print("Error with partial query:", e)

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


agent_tools_list = [
    GetAllAircraftInfo,
    GetConflictInfo,
    SendCommand,
    ContinueMonitoring,
    QueryConflicts,
]


agent_tool_dict = {tool.name: tool for tool in agent_tools_list}
