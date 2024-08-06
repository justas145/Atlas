# %%
from agent_tools import (
    initialize_client,
    initialize_collection,
    agent_tools_list,
    GetConflictInfo,
    receive_bluesky_output
)
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
from langchain_openai import ChatOpenAI

sys.path.append("../bluesky")
from bluesky.network.client import Client

from contextlib import contextmanager
import sys
import io

import logging
import os
import threading
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

# Connect to BlueSky simulator
client = Client()
client.connect("127.0.0.1", 11000, 11001)
client.update()
# Initialize the client in agent_tools
initialize_client(client)


# Setup basic logging
# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
# )


# Define a custom logging handler
# class InMemoryLogHandler(logging.Handler):
#     def __init__(self):
#         super().__init__()
#         self.logs = []

#     def emit(self, record):
#         # Here you could process the log record, e.g., check if it's an error, extract information, etc.
#         log_entry = self.format(record)
#         self.logs.append(log_entry)  # Store log entries in a list
#         print(log_entry)  # Optionally print the log entry


# # Configure logging to use the custom handler
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# handler = InMemoryLogHandler()
# formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
# handler.setFormatter(formatter)
# logger.addHandler(handler)


def log_monitor():
    """Function to simulate log generation, replace this with actual logging from your application"""
    while True:
        logging.info("Simulated log entry.")
        time.sleep(5)  # Simulate logging every 5 seconds


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


# Setup basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_messages = []

    def emit(self, record):
        self.log_messages.append(self.format(record))


def monitor_too_many_requests():
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
                #print_output("Too many requests, pausing operations...")
                client.send_event(b"STACK", "HOLD")
                list_handler.log_messages.clear()  # Clear the log messages after handling
                break


def monitor_crashes(callback):
    path_to_watch = "../bluesky/output"  # Make sure this is correct
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


# %%
dotenv.load_dotenv("../.env")

# llama3-70b-8192
# llama3-groq-70b-8192-tool-use-preview
temperature = 0.3
model_name = "llama3-70b-8192"
# llama-3.1-70b-versatile

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


# %%
# Vector DB
collections = chroma_client.list_collections()
collection_names = [collection.name for collection in collections]
selected_collection = "experience_library_v1"

collection = chroma_client.get_or_create_collection(
    name=selected_collection,
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"},
)
# Initialize the collection in agent_tools
initialize_collection(collection)


# %%

prompt = hub.pull("hwchase17/openai-tools-agent")

with open("prompts/system_with_exp_lib.txt", "r") as f:
    prompt.messages[0].prompt.template = f.read()

# with open("prompts/conflict.txt", "r") as f:
#     prompt.messages[0].prompt.template += f.read()

del agent_tools_list[1]
del agent_tools_list[-1]

agent_tools_list = agent_tools_list


chat = ChatGroq(temperature=temperature, model_name=model_name, api_key=os.getenv("GROQ_API_KEY_3"))
chat = ChatOpenAI(temperature=0.3, model="gpt-4o")


agent = agents.create_openai_tools_agent(chat, agent_tools_list, prompt)

agent_executor = agents.AgentExecutor(
    agent=agent, tools=agent_tools_list, verbose=True)

# %%
user_input = "You are an air traffic controller with tools. Solve aircraft conflict. Solve until there are no more conflicts. "

scenario = "case3"
# ac_3_no_dH_head-on_3
# ac_3_dH_t-formation_8
# ac_3_dH_head-on_9
# ac_3_no_dH_converging_2
# ac_3_no_dH_parallel_4
# ac_4_dH_head - on_10
# ac_4_dH_t-formation_7
# ac_4_no_dH_converging_5
# ac_4_no_dH_parallel_1
# ac_4_no_dH_t-formation_1
# ac_4_no_dH_t-formation_3
client.send_event(b"STACK", f"IC TEST/Big/ac_4/no_dH/t-formation_3.scn")
time.sleep(1.5)
useless = client.update()
useless =  receive_bluesky_output()

def run_ai():
    with CaptureAndPrintConsoleOutput() as output:
        out = agent_executor.invoke({"input": user_input})
        console_output = output.getvalue()
        print("Captured Console Output:")
        print(type(console_output))
        return console_output

if __name__ == "__main__":
    # Run the crash monitoring in a background thread
    threading.Thread(target=monitor_crashes, args=(print_output,), daemon=True).start()
    threading.Thread(target=monitor_too_many_requests, daemon=True).start()
    # client.send_event(b"STACK", "HOLD")
    # time.sleep(10)

    # Run the AI operations in the main thread
    run_ai()
