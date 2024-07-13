from langchain.agents import tool, initialize_agent, AgentType, Tool
from langchain.agents import tool
from langchain_community.llms import Ollama
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from voice_assistant.api_key_manager import get_transcription_api_key, get_response_api_key, get_tts_api_key
from voice_assistant.config import Config
from voice_assistant.utils import delete_file
from voice_assistant.text_to_speech import text_to_speech
from voice_assistant.response_generation import generate_response
from voice_assistant.transcription import transcribe_audio
from voice_assistant.audio import record_audio, play_audio
from colorama import Fore, init
import logging
import pandas as pd
import csv
import json
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
import requests
import argparse
import sys
import os
sys.path.append('../bluesky')  # Adjust the path as neces
sys.path.append(os.path.abspath('../'))
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, create_json_chat_agent, create_structured_chat_agent, create_openai_functions_agent
from langchain_openai import OpenAI, ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from bluesky.network.client import Client
from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import time
from contextlib import contextmanager
from io import StringIO
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents.format_scratchpad.openai_tools import (

    format_to_openai_tool_messages,
)



# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize colorama
init(autoreset=True)


load_dotenv(find_dotenv())


parser = argparse.ArgumentParser(
    description='Run the ChatGrok model with specified parameters.')
parser.add_argument('--temperature', type=float, default=0.3,
                    help='Set the temperature for the model')
parser.add_argument('--model_name', type=str,
                    default="llama3-70b-8192", help='Specify the model name')
parser.add_argument('--system_prompt', type=str,
                    default='You are a helpful assistant' , help='Specify the system prompt')
args = parser.parse_args()

if 'gpt' in args.model_name:
    chat = ChatOpenAI(temperature=args.temperature, model=args.model_name)
else:
    chat = ChatGroq(temperature=args.temperature, model_name=args.model_name)



summary_prompt = PromptTemplate.from_template(
    "Summarise the intermediate steps in first person, keep it short, and for commands write a full command syntax. intermediate steps: {intermediate_steps_str}"
)
summarise_llm_chain = summary_prompt | ChatGroq(temperature=0.3, model_name="llama3-70b-8192") | StrOutputParser()

# Initialization vector db
# get current working dir path

# Get the directory of the current script
base_path = os.path.dirname(__file__)
# Set the path to vectordb relative to the script's location
vectordb_path = os.path.join(base_path, 'skills-library', 'vectordb')
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-large"
)
chroma_client = chromadb.PersistentClient(path=vectordb_path)

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

# Select Vector DB
collections = chroma_client.list_collections()
collection_names = [collection.name for collection in collections]
selected_collection = 'test2'

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



tools = [GetAllAircraftInfo, GetConflictInfo,
         SendCommand, QueryDatabase, ContinueMonitoring]



def get_intermediate_steps(data):
    steps_summary = []
    if 'intermediate_steps' in data:
        for step in data['intermediate_steps']:
            action = step[0].tool
            command = json.dumps(step[0].tool_input)
            # Extracting the response message from log
            invoke = step[0].log
            response = step[1]

            summary = f"Action: {action}, Command: {command}, Invoke: {invoke}, Response: {response}"
            steps_summary.append(summary)

    return '\n\n'.join(steps_summary)



# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-tools-agent")
prompt.messages[0].prompt.template = args.system_prompt
# Create an agent executor by passing in the agent and tools
agent = create_openai_tools_agent(chat, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, return_intermediate_steps=True)



chat_history = []
try:
    while True:
        # Get input from the user
        #user_input = input("Enter your request or press CTRL+C to quit: ")
        
        record_audio("user_input.wav")
        transcription_api_key = get_transcription_api_key()
        user_input = transcribe_audio(
            Config.TRANSCRIPTION_MODEL, transcription_api_key, 'user_input.wav', Config.LOCAL_MODEL_PATH)
        logging.info(Fore.GREEN + "You said: " + user_input)

        # Process the input or pass it to your agent
        # For example, you could have your agent process this input:
        # response = agent.process_input(user_input)
        # print(response)

        # Placeholder for processing input
        print(f"Processing your input: {user_input}")

        out = agent_executor.invoke(
            {"input": user_input, "chat_history": chat_history})
        response_text = out['output']
        intermediate_steps_str = get_intermediate_steps(out)
        print(intermediate_steps_str)
        # itermediate_steps_summary = summarise_llm_chain.invoke({"intermediate_steps_str": intermediate_steps_str})
        # print(itermediate_steps_summary)
        chat_history.append(HumanMessage(content=user_input))
        # chat_history.append(AIMessage(content=itermediate_steps_summary))
        chat_history.append(AIMessage(content=response_text))
    
        if Config.TTS_MODEL == 'openai':
            output_file = 'output.mp3'
        else:
            output_file = 'output.wav'
            
        tts_api_key = get_tts_api_key()
        # Convert the response text to speech and save it to the appropriate file
        text_to_speech(Config.TTS_MODEL, tts_api_key,
                       response_text, output_file, Config.LOCAL_MODEL_PATH)
        
        # Play the generated speech audio
        play_audio(output_file)

        # Clean up audio files
        delete_file('test.wav')
        delete_file(output_file)
except KeyboardInterrupt:
    # Handle the CTRL+C gracefully
    print("\nExiting the application. Goodbye!")
    sys.exit(0)





