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


chat = ChatGroq(temperature=temperature, model_name=model_name)

agent = agents.create_openai_tools_agent(chat, agent_tools_list, prompt)

agent_executor = agents.AgentExecutor(
    agent=agent, tools=agent_tools_list, verbose=True)


# %%
user_input = "solve aircraft conflict with heading changes"

out = agent_executor.invoke({"input": user_input})

# %%
# chat_history = []
# out = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
# chat_history.append(messages.HumanMessage(content=user_input))
# chat_history.append(messages.AIMessage(content=out["output"]))
