# %%
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.prompts import PromptTemplate
import chromadb.utils.embedding_functions as embedding_functions
import chromadb
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import tool
from langchain_openai import ChatOpenAI
from openai import OpenAI
from io import StringIO
from contextlib import contextmanager
import time
from dotenv import load_dotenv, find_dotenv
import os
from bluesky.network.client import Client
import bluesky
import sys
sys.path.append('../../bluesky')  # Adjust the path as necessary

# Now you can import bluesky modules

# %%


# %%
# Load the .env file
load_dotenv(find_dotenv())

# %%
# capture output information from the bluesky client and return it as a string


@contextmanager
def capture_stdout():
    new_stdout = StringIO()
    old_stdout = sys.stdout
    sys.stdout = new_stdout
    try:
        yield new_stdout
    finally:
        sys.stdout = old_stdout


def update_until_complete(client):
    complete_output = ""
    while True:
        with capture_stdout() as captured:
            client.update()
        new_output = captured.getvalue()
        if not new_output.strip():
            # No more new output, break the loop
            break
        complete_output += new_output
    client.update()
    return complete_output


# %%
# connect to client
client = Client()
client.connect("127.0.0.1", 11000, 11001)
client.update()

# %%
client.update()

# %%
# # example commands
# command = 'CRE BZB994 A320 -15.101549,-14.504519 48 FL150 150'
# command = 'GETACIDS'
# #command = 'PAN BZB994'
# # sending command to the client. can be a plugin command or a pre-defined command
client.send_event(b'STACK', 'A10')

# %%
sim_output = update_until_complete(client)
print("All captured output:\n", sim_output)
print('-------------------')

# %%
client.update()

# %%
# initialize the GPT-3.5 model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# %%


@tool
def get_all_aircraft_info(command: str) -> str:
    """Returns each aircraft information. Use command GETACIDS to get all aircrafts."""
    client.send_event(b'STACK', command)
    sim_output = update_until_complete(client)
    return sim_output


# %%
@tool
def send_command_to_simulator(command: str, args: list = None) -> str:
    """Sends a command with optional arguments to the simulator and returns the output.
    Parameters:
    - command (str): The command to send to the simulator.

    - args (list, optional): A list of arguments to include with the command. Defaults to None.

    Returns:

    str: The output from the simulator.
    """

    # Ensure args is a list if not provided

    if args is None:

        args = []

    else:
        args = [str(item) for item in args]

    # Convert the command and its arguments into a byte string to be sent

    # Note: Assuming that arguments are separated by spaces in the command string

    command_with_args = ' '.join([command] + args)

    # Send the command to the simulator
    # client.update()
    client.send_event(b'STACK', command_with_args)

    # wait 1 second
    time.sleep(1)

    # Wait for and retrieve the output from the simulator

    sim_output = update_until_complete(client)

    return sim_output

# %%


tools = [send_command_to_simulator]

# %%
# Create a chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, that can also use tools to answer questions",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# %%
llm_with_tools = llm.bind_tools(tools)

# %%

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

# %%

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# %%

input1 = "Create an airraft for me. it should be a A320 with the A10 callsign, at -15.101549,-14.504519, at 48 heading, flying with the speed of 150 at flight level150."

input1 = "Are there any potential conflicts between aircrafts?"


input2 = "Hey agent, please move the A10 aircraft to the position of longitude -51.604803 and latitude -40.405723"

input3 = "Please change heading of A10 aircraft to 180 degrees."

input4 = "Delete A10 aircraft. I hate this aircraft and don't want to see it anymore. After you delete it, please verify if the aircraft still there. you can verify it by sending the command ACID, which in this case is A10. If the aircraft information is still there, then it means the aircraft is not deleted."


# %% [markdown]
# # Vector database integration

# %%

# %%
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-large"
)

# %%
vectordb_path = 'C:/Users/justa/OneDrive/Desktop/Developer/LLM-Enhanced-ATM/llm/skills-library/vectordb'
chroma_client = chromadb.PersistentClient(path=vectordb_path)


# %%
# Get a collection object from an existing collection, by name. If it doesn't exist, create it.
collection = chroma_client.get_or_create_collection(
    name="test1", embedding_function=openai_ef, metadata={"hnsw:space": "cosine"})

# %%
collection.peek()

# %%

prompt_template = PromptTemplate.from_template(
    "{input} \n Below is some documentation that can either be useful or not for you to use in order to generate and send a correct command to the simulator. \n\n {doc1} \n {doc2} \n {doc3}."
)

# %%
query_results1 = collection.query(
    query_texts=[input1],
    n_results=5
)

# %%
query_results1

# %%
query_results2 = collection.query(
    query_texts=[input2],
    n_results=3
)

# %%
query_results3 = collection.query(
    query_texts=[input3],
    n_results=3
)

# %%
query_results4 = collection.query(
    query_texts=[input4],
    n_results=3
)

# %%
query_results4['documents'][0]

# %%
final_input1 = prompt_template.format(
    input=input1, doc1=query_results1['documents'][0][0], doc2=query_results1['documents'][0][1], doc3=query_results1['documents'][0][1])

# %%
final_input2 = prompt_template.format(
    input=input2, doc1=query_results2['documents'][0][0], doc2=query_results2['documents'][0][1], doc3=query_results2['documents'][0][1])

# %%
final_input3 = prompt_template.format(
    input=input3, doc1=query_results3['documents'][0][0], doc2=query_results3['documents'][0][1], doc3=query_results3['documents'][0][1])

# %%
final_input4 = prompt_template.format(
    input=input4, doc1=query_results4['documents'][0][0], doc2=query_results4['documents'][0][1], doc3=query_results4['documents'][0][1])

# %%
out_lst = list(agent_executor.stream({"input": final_input1}))

# %%
out_lst = list(agent_executor.stream({"input": final_input2}))

# %%
out_lst = list(agent_executor.stream({"input": final_input3}))


# %%
final_input4

# %%
out_lst = list(agent_executor.stream({"input": final_input4}))

# %%
# TODO
# 1. connect to a memory block so it will be able to remember the previous commands and responses
# 2. create a vector database with commands and their descriptions (just like voyager did). Input is a prompt, output is top 5 commands
# 3. Make a system that makes the agent write its own commands and test out the commands


# %% [markdown]
# # Memory Part (not finished)

# %%

MEMORY_KEY = "chat_history"
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant",
        ),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# %%

chat_history = []

# %%
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

# %%
chat_history = []
input1 = "how many aircraft are there in the sky currently?"

result1 = agent_executor.invoke(

    {"input": input1, "chat_history": chat_history})

# chat history should have human message, output result from sim, and output result from openai

# %%
intermediate_steps1 = result1['intermediate_steps'][0][1]

chat_history.extend(
    [

        HumanMessage(content=input1),
        AIMessage(content=result1["output"]),
        AIMessage(content=intermediate_steps1)
    ]

)
# order here matters for some reason. If the order is changed, for the next question the tool will be invoke even though it is not needed

# %%
chat_history

# %%
input2 = "and which aircraft has the lowest altitude?"
result2 = agent_executor.invoke({"input": input2,
                                 "chat_history": chat_history})

# %%
try:
    intermediate_steps2 = result2['intermediate_steps'][0][1]
except IndexError:
    intermediate_steps2 = ''


# %%

chat_history.extend(
    [

        HumanMessage(content=input2),
        AIMessage(content=result2["output"]),
        AIMessage(content=intermediate_steps2)
    ]

)
# order here matters for some reason. If the order is changed, for the next question the tool will be invoke even though it is not needed

# %%
chat_history

# %%
input3 = "What is the aircraft id with the lowest altitude?"
result3 = agent_executor.invoke({"input": input3,
                                 "chat_history": chat_history})

# %%
input4 = "What is the alititude of BBB999?"
result4 = agent_executor.invoke({"input": input4,
                                 "chat_history": chat_history})

# %%
result4['intermediate_steps'][0][1]
