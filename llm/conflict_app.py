
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_community.chat_models import ChatOllama
from langchain_openai import OpenAI, ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent, create_json_chat_agent, create_structured_chat_agent, create_openai_functions_agent
from langchain import hub
import sys
sys.path.append('../bluesky')  # Adjust the path as necessary
from bluesky.network.client import Client
import streamlit as st
from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from prompts.prompts import conflict_prompt
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv, find_dotenv
import time
from contextlib import contextmanager
from io import StringIO
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents.format_scratchpad.openai_tools import (

    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage

import os
from langchain_community.llms import Ollama
from langchain.agents import tool
from langchain.agents import tool, initialize_agent, AgentType, Tool
import sys
import requests

load_dotenv(find_dotenv())

# Initialization
vectordb_path = 'C:/Users/justa/OneDrive/Desktop/Developer/LLM-Enhanced-ATM/llm/skills-library/vectordb'
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-large"
)
chroma_client = chromadb.PersistentClient(path=vectordb_path)

# capture output information from the bluesky client and return it as a string


st_callback = StreamlitCallbackHandler(st.container())

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



# Streamlit UI
st.title("LLM-Enhanced ATM Simulation")

# Select Vector DB
collections = chroma_client.list_collections()
collection_names = [collection.name for collection in collections]
selected_collection = st.selectbox(
    "Select a Vector Database", collection_names)

collection = chroma_client.get_or_create_collection(
    name=selected_collection, embedding_function=openai_ef, metadata={"hnsw:space": "cosine"})



# Connect to Simulator

client = Client()
client.connect("127.0.0.1", 11000, 11001)
client.update()
client.update()
    
### Tools ###


# get all aircraft info
# get conflict information

@tool
def GetAllAircraftInfo(command: str = 'GETACIDS'):
    """Get each aircraft information at current time: position, heading (deg), track (deg), altitude, V/S (vertical speed), calibrated, true and ground speed and mach number. Input is 'GETACIDS'.
    
    Parameters:
    - command: str (default 'GETACIDS')
    
    Example usage:
    - GetAllAircraftInfo('GETACIDS')
    
    Returns:
    - str: all aircraft information
    """
    command = command.replace('"', '').replace("'", "")
    command = command.split('\n')[0]
    print(f'LLM input:{command}')

    client.send_event(b'STACK', command)
    time.sleep(1)
    sim_output = update_until_complete(client)
    return sim_output


@tool
def GetConflictInfo(commad: str = 'SHOWTCPA'):
    """Use this tool to identify and get vital information on aircraft pairs in conflict. It gives you Time to Closest Point of Approach (TCPA), Quadrantal Direction (QDR), separation distance, Closest Point of Approach distance (DCPA), and Time of Loss of Separation (tLOS).
    
    Parameters:
    - command: str (default 'SHOWTCPA')
    
    Example usage:
    - GetConflictInfo('SHOWTCPA')
    
    Returns:
    - str: conflict information between aircraft pairs
    """
    client.send_event(b'STACK', 'SHOWTCPA')
    client.send_event(b'STACK', 'GETACIDS')
    time.sleep(1)
    sim_output = update_until_complete(client)
    return sim_output


@tool
def ContinueMonitoring(duration: str = '5'):
    """Monitor for conflicts between aircraft pairs for a specified time. 
    Parameters:
    - time (str): The time in seconds to monitor for conflicts. Default is 5 seconds.
    
    Example usage:
    - ContinueMonitoring('5')
    
    Returns:
    - str: The conflict information between aircraft pairs throughout the monitoring period.
    """
    sim_output = ''
    for i in range(int(duration)):
        client.send_event(b'STACK', 'SHOWTCPA')
        time.sleep(1)
        sim_output += str(i) + ' sec: \n' + \
            update_until_complete(client) + '\n'
    return sim_output


@tool
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
    # Convert the command and its arguments into a string to be sent
    # command_with_args = ' '.join([command] + [str(arg) for arg in args])
    # Send the command to the simulator
    # client.update()  # Uncomment this if you need to update the client state before sending the command
    print(command)
    # replace " or ' in the command string with nothing
    command = command.replace('"', '').replace("'", "")
    command = command.split('\n')[0]
    client.send_event(b'STACK', command)
    # wait 1 second
    time.sleep(1)
    # Wait for and retrieve the output from the simulator
    sim_output = update_until_complete(client)
    if sim_output == '':
        return 'Command executed successfully.'
    if 'Unknown command' in sim_output:
        return sim_output + '\n' + 'Please use a tool QueryDatabase to search for the correct command.'
    return sim_output


@tool
def QueryDatabase(input: str):
    """If you want to send command to a simulator please first search which command you should use. For example if you want to create an aircraft, search for 'how do I create an aircraft'.
    Parameters:
    - input: str (the query to search for)
    Returns:
    - list: the top 5 results from the database
    """

    query_results = collection.query(
        query_texts=[input],
        n_results=5
    )
    return query_results
    return "The HDG command sets the aircraft's heading, disengaging LNAV mode. Use it by specifying the aircraft ID and desired heading in degrees, like: HDG acid, hdg_degrees. The aircraft ID is the unique identifier of the aircraft you want to control, and the heading is the desired direction in degrees. For example, to set the heading of aircraft ABC to 90 degrees, you would use the command: HDG ABC, 90 \n\n\n The ALT command adjusts an aircraft's altitude via autopilot, optionally setting vertical speed. Specify the aircraft ID, desired altitude in feet, and optionally, climb/descent speed in feet per minute, like ALT acid, alt, vspd. For example if you want to change aircraft KL123 height to 20000 ft the command is: ALT KL123 20000. or ALT KL123 FL200, or ALT KL123 FL200 10"
    # return query_results['documents'][0]


tools = [GetAllAircraftInfo, GetConflictInfo,
         SendCommand, QueryDatabase, ContinueMonitoring]


def invoke_until_success(llm):
    while True:
        try:
            # Attempt to invoke the method
            llm.invoke('hi')
            print("Invocation successful.")
            break  # Exit the loop if invocation was successful
        except Exception as e:
            print("Invocation failed, trying again...")
            time.sleep(1)  # Wait for a short period before retrying

def fetch_model_names(base_url):
    api_endpoint = f"{base_url}/api/tags"
    try:
        response = requests.get(api_endpoint)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        data = response.json()
        model_names = [model['name'] for model in data.get('models', [])]
        return model_names
    except requests.RequestException as e:
        return str(e)
    

def load_non_gpt_model(model_name):
    llm = ChatOllama(base_url="https://ollama.junzis.com", model=model_name)
    invoke_until_success(llm)
    return llm


# User Input
user_input = st.text_area(
    "Enter your input", value="Check if there are any conflicts between aircraft and if there are resolve them by changing altitude or heading. Your Goal is to have no conflicts between aircraft.")

# Select Model
model_names = fetch_model_names("https://ollama.junzis.com")

# add gpt model names to the list
gpt_models = ['gpt-4-turbo-2024-04-09', 'gpt-3.5-turbo-0125', 'gpt-3.5-turbo-instruct']
model_names.extend(gpt_models)

model_name = st.selectbox("Select a Model", model_names)

# Select Model Type
model_types = ["react", "structured", "openai"]
model_type = st.selectbox("Select a Model Type", model_types)

# Initialize LLM based on selection
# Function to load non-GPT models with retries



# Model selection dropdown

# Check for the model in the session state or load it if not present or if model name has changed
if 'llm' not in st.session_state or 'model_name' not in st.session_state or st.session_state['model_name'] != model_name:
    if model_name not in gpt_models:
        # Load non-GPT model if not in session state or if a different model is selected
        st.session_state['llm'] = load_non_gpt_model(model_name)
    elif "gpt" in model_name:
        # Load GPT model if applicable and not in session state or if a different model is selected
        st.session_state['llm'] = ChatOpenAI(model=model_name)
    # Update the session state to reflect the current model
    st.session_state['model_name'] = model_name

# Retrieve the model from session state
llm = st.session_state['llm']


# Scenario File Selection
scenario = st.selectbox("Select a Scenario File", [
                        "case1", "case2", "case3", "case4", "case5", "case6"])

# Agent Execution based on Model Type
if st.button("Run"):
    if model_type == "react":
        react_prompt = hub.pull("hwchase17/react")
        user_template = st.text_input(
            "Modify the Template", value=react_prompt.template)
        react_prompt.template = user_template
        react_agent = create_react_agent(llm, tools, react_prompt)
        agent_executor = AgentExecutor(
            agent=react_agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=70)
    elif model_type == "structured":
        structured_prompt = hub.pull("hwchase17/structured-chat-agent")
        structured_prompt.messages[0].prompt.template = 'Respond to the human as helpfully and accurately as possible. You have access to the following tools:\n\n{tools}\n\nUse a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).\n\nValid "action" values: "Final Answer" or {tool_names}\n\nProvide only ONE action per $JSON_BLOB, as shown:\n\n```\n{{\n  "action": $TOOL_NAME,\n  "action_input": $INPUT\n}}\n```\n\nFollow this format:\n\nQuestion: input question to answer\nThought: consider previous and subsequent steps\nAction:\n```\n$JSON_BLOB\n```\nObservation: action result\n... (repeat Thought/Action/Observation N times)\nThought: I know what to respond\nAction:\n```\n{{\n  "action": "Final Answer",\n  "action_input": "Final response to human"\n}}\n```\nBegin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation'
        structured_agent = create_structured_chat_agent(
            llm, tools, structured_prompt)
        agent_executor = AgentExecutor(
            agent=structured_agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=70)
    elif model_type == "openai":
        openai_function_prompt = hub.pull("hwchase17/openai-functions-agent")
        openai_function_agent = create_openai_functions_agent(
            llm, tools, openai_function_prompt)
        agent_executor = AgentExecutor(
            agent=openai_function_agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=70)

    client.send_event(b'STACK', f'IC simple/conflicts/2ac/{scenario}.scn')
    # Assume update_until_complete is defined elsewhere
    time.sleep(1)
    update_until_complete(client)
    update_until_complete(client)

    with tracing_v2_enabled(project_name="My Project", tags=[model_name, scenario, model_type]):
        # response = agent_executor.invoke({"input": user_input}, {"callbacks": [st_callback]})
        # st.write(response["output"])  # Display the output in Streamlit

        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            response = agent_executor.invoke(
                {"input": user_input}, {"callbacks": [st_callback]}
            )
            st.write(response["output"])