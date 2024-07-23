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
from utils.skill_manual_creation import create_skill_manual, update_skill_library
import chromadb
import dotenv
from langchain import agents, hub
from langchain_core import messages
from langchain_core.tools import tool as langchain_tool
from langchain_groq import ChatGroq

sys.path.append("../bluesky")
from bluesky.network.client import Client

from contextlib import contextmanager
import sys
import io


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

chat = ChatGroq(temperature=temperature, model_name=model_name, api_key=os.getenv("GROQ_API_KEY"))

agent = agents.create_openai_tools_agent(chat, agent_tools_list, prompt)

agent_executor = agents.AgentExecutor(
    agent=agent, tools=agent_tools_list, verbose=True)


# %%
user_input = "You are an air traffic controller with tools. Solve aircraft conflict. Solve until there are no more conflicts"

scenario = "case1"
client.send_event(b"STACK", f"IC simple/conflicts/2ac/{scenario}.scn")
time.sleep(1.5)
client.update()

with CaptureAndPrintConsoleOutput() as output:
    out = agent_executor.invoke({"input": user_input})

console_output = output.getvalue()
print("Captured Console Output:")
print(type(console_output))


# skill_manual, metadata = create_skill_manual(console_output)
# update_skill_library(collection, skill_manual, metadata)

final_conflict_check = GetConflictInfo("SHOWTCPA")
print(final_conflict_check)
