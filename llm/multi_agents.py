# %%
from agent_tools import (
    initialize_client,
    initialize_collection,
    agent_tools_list,
    agent_tool_dict
)
import os
import sys
import time
from contextlib import contextmanager
from io import StringIO

import chromadb
import dotenv
from langchain import agents, hub
from langchain.agents import (
    AgentExecutor,
    create_structured_chat_agent,
    create_react_agent,
    create_openai_tools_agent,
)
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage

from langchain_core import messages
from langchain_core.tools import tool as langchain_tool
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
sys.path.append("../bluesky")
from bluesky.network.client import Client
import json
from prompts.agent_prompts import (
    planner_prompt,
    executor_prompt,
    verifier_prompt,
    all_prompt,
)

# %%
dotenv.load_dotenv("../.env")

temperature = 0.4
model_name = "Llama3-70b-8192"
chat = ChatGroq(temperature=temperature, model_name=model_name)

#chat = ChatOpenAI(model='gpt-4o', temperature=0.1)

# %%
# Initialization vector db

base_path = os.path.dirname(__file__)
vectordb_path = os.path.join(base_path, "skills-library", "vectordb")
prompts_path = os.path.join(base_path, "prompts", "prompts.py")

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
print("Connected to BlueSky")

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


def get_intermediate_steps(data):
    steps_summary = []
    if "intermediate_steps" in data:
        for step in data["intermediate_steps"]:
            action = step[0].tool
            command = json.dumps(step[0].tool_input)
            # Extracting the response message from log
            invoke = step[0].log
            response = step[1]

            summary = f"Action: {action}, Command: {command}, Invoke: {invoke}, Response: {response}"
            steps_summary.append(summary)

    return "\n\n".join(steps_summary)


# chat = ChatOpenAI(temperature=0.2, model_name='gpt-4o')
# chat = ChatOllama(
#     base_url="https://ollama.junzis.com", model="mixtral:8x22b", temperature=0.2
# )
agent_tools_list_planner = [agent_tool_dict[tool] for tool in ["GETALLAIRCRAFTINFO", "GETCONFLICTINFO"]]
agent_tools_list_controller = [agent_tool_dict[tool] for tool in ["SENDCOMMAND"]]
agent_tools_list_verifier = [
    agent_tool_dict[tool] for tool in ["GETALLAIRCRAFTINFO", "GETCONFLICTINFO", "CONTINUEMONITORING"]
]


system_prompt = hub.pull("hwchase17/openai-tools-agent")


agent_planner = create_openai_tools_agent(chat, agent_tools_list_planner, system_prompt)
agent_executor_planner = AgentExecutor(
    agent=agent_planner,
    tools=agent_tools_list_planner,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
)

agent_controller = create_openai_tools_agent(
    chat, agent_tools_list_controller, system_prompt
)
agent_executor_controller = AgentExecutor(
    agent=agent_controller,
    tools=agent_tools_list_controller,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
)


agent_verifier = create_openai_tools_agent(
    chat, agent_tools_list_verifier, system_prompt
)
agent_executor_verifier = AgentExecutor(
    agent=agent_verifier,
    tools=agent_tools_list_verifier,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
)
# %%
# planner

with open("prompts/ICAO_seperation_guidelines.txt", "r") as f:
    icao_seperation_guidelines = f.read()


planner_prompt = planner_prompt.format(
    icao_seperation_guidelines=icao_seperation_guidelines
)
scenario = 'case7'
client.send_event(b'STACK', f'IC simple/conflicts/2ac/{scenario}.scn')
time.sleep(1.5)
client.update()


planner_output_lst = []
planner_intermediate_steps_lst = []
controller_output_lst = []
controller_intermediate_steps_lst = []
verifier_output_lst = []
verifier_intermediate_steps_lst = []

planner_output = agent_executor_planner.invoke({"input": planner_prompt})
input = planner_output["input"]
plan = planner_output["output"]
planner_output_lst.append(plan)
planner_intermediate_steps_lst.append(
    get_intermediate_steps(planner_output)
)
if "no conflicts" in plan:
    print(plan)
    exit()
while True:
    controller_prompt = executor_prompt.format(plan=plan)
    controller_output = agent_executor_controller.invoke({"input": controller_prompt})
    controller_output_lst.append(controller_output["output"])
    controller_intermediate_steps_lst.append(get_intermediate_steps(controller_output))

    verifier_prompt = verifier_prompt.format(
        icao_seperation_guidelines=icao_seperation_guidelines, plan=plan
    )
    verifier_output = agent_executor_verifier.invoke({"input": verifier_prompt})
    verifier_output_lst.append(verifier_output["output"])
    verifier_intermediate_steps_lst.append(get_intermediate_steps(verifier_output))
    replan = verifier_output["output"]
    if "no conflicts" in replan.lower():
        print("TASK:")
        print(input)
        print("Planner Output:")
        print("\n".join(planner_output_lst))
        print("Planner Intermediate Steps:")
        print("\n".join(planner_intermediate_steps_lst))
        print("Controller Output:")
        print("\n".join(controller_output_lst))
        print("Controller Intermediate Steps:")
        print("\n".join(controller_intermediate_steps_lst))
        print("Verifier Output:")
        print("\n".join(verifier_output_lst))
        print("Verifier Intermediate Steps:")
        print("\n".join(verifier_intermediate_steps_lst))

        exit()
    else:
        plan = replan
