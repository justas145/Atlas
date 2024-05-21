# %%
import os
import sys
import time
from contextlib import contextmanager
from io import StringIO

import chromadb
import dotenv
from langchain import agents, hub
from langchain_core import messages
from langchain_core.tools import tool as langchain_tool
from langchain_groq import ChatGroq

sys.path.append("../bluesky")
from bluesky.network.client import Client

# %%
dotenv.load_dotenv("../.env")

temperature = 0.3
model_name = "llama3-70b-8192"


# %%
# Initialization vector db

base_path = os.path.dirname(__file__)
vectordb_path = os.path.join(base_path, "skills-library", "vectordb")

chroma_client = chromadb.PersistentClient(path=vectordb_path)

openai_ef = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-large",
)


# %%
# Connect to BlueSky simulator
client = Client()
client.connect("127.0.0.1", 11000, 11001)
client.update()


# %%
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


# %%
# Vector DB
collections = chroma_client.list_collections()
collection_names = [collection.name for collection in collections]
selected_collection = "test2"

collection = chroma_client.get_or_create_collection(
    name=selected_collection,
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"},
)


# %%
### Tools ###


@langchain_tool
def GetAllAircraftInfo(command: str = "GETACIDS"):
    """Get each aircraft information at current time: position, heading (deg), track (deg), altitude, V/S (vertical speed), calibrated, true and ground speed and mach number. Input is 'GETACIDS'.

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

    client.send_event(b"STACK", command)
    time.sleep(0.1)
    sim_output = receive_bluesky_output()
    return sim_output


@langchain_tool
def GetConflictInfo(commad: str = "SHOWTCPA"):
    """Use this tool to identify and get vital information on aircraft pairs in conflict. It gives you Time to Closest Point of Approach (TCPA), Quadrantal Direction (QDR), separation distance, Closest Point of Approach distance (DCPA), and Time of Loss of Separation (tLOS).

    Parameters:
    - command: str (default 'SHOWTCPA')

    Example usage:
    - GetConflictInfo('SHOWTCPA')

    Returns:
    - str: conflict information between aircraft pairs
    """
    client.send_event(b"STACK", "SHOWTCPA")
    time.sleep(0.1)
    sim_output = receive_bluesky_output()
    return sim_output


@langchain_tool
def ContinueMonitoring(duration: int = 5):
    """Monitor for conflicts between aircraft pairs for a specified time.
    Parameters:
    - time (int): The time in seconds to monitor for conflicts. Default is 5 seconds.

    Example usage:
    - ContinueMonitoring(5)

    Returns:
    - str: The conflict information between aircraft pairs throughout the monitoring period.
    """
    time.sleep(duration)
    client.send_event(b"STACK", "SHOWTCPA")
    time.sleep(0.1)
    sim_output = receive_bluesky_output()
    return sim_output


@langchain_tool
def SendCommand(command: str):
    """
    Sends a command with optional arguments to the simulator and returns the output.
    You can only send 1 command at a time.

    Parameters:
    - command (str): The command to send to the simulator. Can only be a single command, with no AND or OR operators.

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
    time.sleep(0.1)
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


@langchain_tool
def QueryDatabase(input: str):
    """If you want to send command to a simulator please first search which command you should use. For example if you want to create an aircraft, search for 'how do I create an aircraft'.
    Parameters:
    - input: str (the query to search for)
    Returns:
    - list: the top 5 results from the database
    """

    query_results = collection.query(query_texts=[input], n_results=5)
    return query_results


agent_tools = [
    GetAllAircraftInfo,
    GetConflictInfo,
    SendCommand,
    QueryDatabase,
    ContinueMonitoring,
]

# %%

prompt = hub.pull("hwchase17/openai-tools-agent")

with open("prompts/system.txt", "r") as f:
    prompt.messages[0].prompt.template = f.read()

with open("prompts/conflict.txt", "r") as f:
    prompt.messages[0].prompt.template += f.read()


chat = ChatGroq(temperature=temperature, model_name=model_name)

agent = agents.create_openai_tools_agent(chat, agent_tools, prompt)

agent_executor = agents.AgentExecutor(agent=agent, tools=agent_tools, verbose=True)


# %%
user_input = "solve aircraft conflict with heading changes"

out = agent_executor.invoke({"input": user_input})

# %%
# chat_history = []
# out = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
# chat_history.append(messages.HumanMessage(content=user_input))
# chat_history.append(messages.AIMessage(content=out["output"]))
