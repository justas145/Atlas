# %%
from agent_tools import (
    initialize_client,
    initialize_collection,
    agent_tools_list,
    GetConflictInfo,
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

# Setup basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
temperature = 1.2
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
# Connect to BlueSky simulator
client = Client()
client.connect("127.0.0.1", 11000, 11001)
client.update()
# Initialize the client in agent_tools
initialize_client(client)


# %%
# Vector DB
collections = chroma_client.list_collections()
collection_names = [collection.name for collection in collections]
selected_collection = "skill_manuals"

collection = chroma_client.get_or_create_collection(
    name=selected_collection,
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"},
)
# Initialize the collection in agent_tools
initialize_collection(collection)


# %%

prompt = hub.pull("hwchase17/openai-tools-agent")

with open("prompts/system.txt", "r") as f:
    prompt.messages[0].prompt.template = f.read()

# with open("prompts/conflict.txt", "r") as f:
#     prompt.messages[0].prompt.template += f.read()
agent_tools_list = agent_tools_list[:-1]


from langchain_together import ChatTogether

# choose from our 50+ models here: https://docs.together.ai/docs/inference-models
# chat = ChatTogether(
#     # together_api_key="YOUR_API_KEY",
#     model="meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
# )
chat = ChatGroq(temperature=temperature, model_name=model_name, api_key=os.getenv("GROQ_API_KEY_2"))
chat = ChatOpenAI(temperature=0.3, model='gpt-4o')
# from langchain_fireworks import Fireworks, ChatFireworks
# from dotenv import load_dotenv, find_dotenv
# import os

# llm = ChatFireworks(
#     api_key=os.getenv("FIREWORKS_API_KEY"),
#     model="accounts/fireworks/models/llama-v3-70b-instruct",
# )


agent = agents.create_openai_tools_agent(chat, agent_tools_list, prompt)

agent_executor = agents.AgentExecutor(
    agent=agent, tools=agent_tools_list, verbose=True)

# %%
user_input = "You are an air traffic controller with tools. Solve aircraft conflict. Solve until there are no more conflicts"

scenario = "case3"
client.send_event(b"STACK", f"IC TEST/Big/ac_3/dH/head-on_10.scn")
time.sleep(1.5)
client.update()

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


    # Run the AI operations in the main thread
    run_ai()
