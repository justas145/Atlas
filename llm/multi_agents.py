# %%
from agent_tools import (
    initialize_client,
    initialize_collection,
    agent_tools_list,
)
import os
import sys
import time
from contextlib import contextmanager
from io import StringIO

import chromadb
import dotenv
from langchain import agents, hub
from langchain.agents import AgentExecutor, create_structured_chat_agent

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
# Initialize the client in agent_tools
initialize_client(client)


# %%
# Vector DB
collections = chroma_client.list_collections()
collection_names = [collection.name for collection in collections]
selected_collection = "default"

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

with open("prompts/conflict.txt", "r") as f:
    prompt.messages[0].prompt.template += f.read()


chat = ChatGroq(temperature=0.2, model_name=model_name)


agent_tools_list_planner = agent_tools_list[0:2]
agent_tool_list_controller = [agent_tools_list[i] for i in (0, 2)]
agent_tool_list_retriever = [agent_tools_list[-1]]
# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/structured-chat-agent")
agent_planner = create_structured_chat_agent(chat, agent_tools_list_planner, prompt)
agent_executor_planner = AgentExecutor(
    agent=agent_planner,
    tools=agent_tools_list_planner,
    verbose=True,
    handle_parsing_errors=True,
)

agent_controller = create_structured_chat_agent(
    chat, agent_tool_list_controller, prompt
)
agent_executor_controller = AgentExecutor(
    agent=agent_controller,
    tools=agent_tool_list_controller,
    verbose=True,
    handle_parsing_errors=True,
)

agent_retriever = create_structured_chat_agent(chat, agent_tool_list_retriever, prompt)
agent_executor_retriever = AgentExecutor(
    agent=agent_retriever,
    tools=agent_tool_list_retriever,
    verbose=True,
    handle_parsing_errors=True,
)

# agent = agents.create_openai_tools_agent(chat, agent_tools_list, prompt)

# agent_executor = agents.AgentExecutor(
#     agent=agent, tools=agent_tools_list, verbose=True)


# %%
# planner
start_input = """
Analyze the conflicts between aircraft, identify the type of conflicts and the seperation requirements from ICAO. As a final answer provide an actionable plan to resolve the conflict. For example:
1. <Aircraft ID> <Action> <Reason and ICAO requirement>.
2. <Aircraft ID> <Action> <Reason and ICAO requirement>.
3. <Aircraft ID> <Action> <Reason and ICAO requirement>.
4. ...
...
"""

planner_output = agent_executor_planner.invoke({"input": start_input})

plan = planner_output["output"]


# commands.txt path is base path + prompts + commands.txt
commands_path = os.path.join(base_path, "prompts", "commands.txt")

# read commands and use base_commands variable
with open(commands_path, "r") as f:
    base_commands = f.read()

retriever_prompt = f"""Find all the necesarry commands that are needed to execute the plan plan.

<Plan>: 
{plan}
<Plan>

<Commands>
{base_commands}
<Commands>

Present your answer in this way:
1. <Command> 
<Description of command including the meanings of the arguments>
<Command Syntax>
2. ...
3. ...
"""

retriever_output = agent_executor_retriever.invoke({"input": retriever_prompt})

bluesky_commands = retriever_output["output"]


controller_prompt = f"""Execute: {plan}
Here are some commands that will help you execute the plan:
{bluesky_commands}
"""
controller_output = agent_executor_controller.invoke({"input": controller_prompt})
