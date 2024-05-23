import matplotlib.pyplot as plt
import yaml
from langchain.chains.openai_functions import create_structured_output_chain
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator
import sys
import os
sys.path.append('../../bluesky')  # Adjust the path as neces
sys.path.append(os.path.abspath('../../'))
from langchain.agents import tool, initialize_agent, AgentType, Tool
from langchain.agents import tool
from langchain_community.llms import Ollama
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents.format_scratchpad.openai_tools import (

    format_to_openai_tool_messages,
)
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from io import StringIO
from contextlib import contextmanager
import time
import chromadb.utils.embedding_functions as embedding_functions
import chromadb
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_experimental.utilities import PythonREPL
from langchain.agents import Tool
from bluesky.network.client import Client
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_community.chat_models import ChatOllama
from langchain_openai import OpenAI, ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent, create_json_chat_agent, create_structured_chat_agent, create_openai_functions_agent
from langchain import hub
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


load_dotenv(find_dotenv())

# Function to read YAML configuration

parser = argparse.ArgumentParser(
    description='Evaluate the chatbot on the bluesky operations dataset.')
parser.add_argument('--config', type=str,
                    help='specify the configuration file path')
parser.add_argument('--test_csv', type=str,
                    help='path to the test csv file')
args = parser.parse_args()

config_path = args.config
csv_input = args.test_csv

# config filename without extension
config_filename = os.path.splitext(os.path.basename(config_path))[0]

# base_path = llm-enhanced-atm/llm
base_path = os.path.dirname(os.path.dirname(__file__))


csv_output = os.path.join(
    base_path, 'data', 'bluesky_operations', 'results', f'{config_filename}.csv')

csv_concatinated = os.path.join(
    base_path, 'data', 'bluesky_operations', 'concatinated_results', f'{config_filename}.csv')

csv_final = os.path.join(base_path, 'data', 'bluesky_operations',
                         'final_results', f'{config_filename}.csv')


def read_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    

config = read_config(config_path)
agent_model_name = config['model']
agent_type = config['agent_type']
system_prompt = config['system_prompt']
temperature = config['temperature']
base_url = config['base_url']



# chat = ChatGroq(temperature=args.temperature, model_name=args.model_name)

summary_prompt = PromptTemplate.from_template(
    "Summarise the intermediate steps in first person, keep it short, and for commands write a full command syntax. intermediate steps: {intermediate_steps_str}"
)
summarise_llm_chain = summary_prompt | ChatGroq(
    temperature=0.3, model_name="llama3-70b-8192") | StrOutputParser()

# Initialization vector db
# get current working dir path

# Get the directory of the current script
# base_path = llm-enhanced-atm/llm
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



# Select Vector DB
collections = chroma_client.list_collections()
collection_names = [collection.name for collection in collections]
selected_collection = 'test1'

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
    time.sleep(0.1)
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
    time.sleep(0.1)
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
        time.sleep(0.1)
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
    time.sleep(0.1)
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

if 'gpt' in agent_model_name:
    agent_model = ChatOpenAI(model=agent_model_name, temperature=temperature)
    print("Agent model is " + agent_model_name)
elif 'llama3-70b-8192' in agent_model_name:
    agent_model = ChatGroq(model=agent_model_name, temperature=temperature)
    print("Agent model is " + agent_model_name)
else:
    agent_model = ChatOllama(
        model=agent_model_name, temperature=temperature, base_url=base_url)
    print("Agent model is " + agent_model_name)

if agent_type == 'openai_tools':
    
    prompt = hub.pull("hwchase17/openai-tools-agent")
    if system_prompt is not None:
        prompt.messages[0].prompt.template = system_prompt
    
    agent = create_openai_tools_agent(agent_model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,
                                handle_parsing_errors=True, return_intermediate_steps=True)
    print("Agent type is " + agent_type)

elif agent_type == 'react':
    prompt = hub.pull("hwchase17/react")
    if system_prompt is not None:
        prompt.template = system_prompt
    agent = create_react_agent(agent_model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,
                                handle_parsing_errors=True, return_intermediate_steps=True)
    print("Agent type is " + agent_type)

else:
    print('Invalid agent type')
    exit()
    



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



def process_csv_inputs(csv_input, csv_output, agent_executor):
    with open(csv_input, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        with open(csv_output, 'a', newline='', encoding='utf-8') as outfile:
            fieldnames = ['user_input', 'output', 'intermediate_steps']
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            if not os.path.isfile(csv_output) or os.path.getsize(csv_output) == 0:
                writer.writeheader()

            for row in reader:
                user_input = row['user_input']
                print('### User Input ###')
                print(user_input)
                scenario = 'A1A2A3'
                client.send_event(
                    b'STACK', f'IC simple/bluesky_operations/{scenario}.scn')
                time.sleep(0.5)  # Wait for the scenario to load
                update_until_complete(client)
                try:
                    out = agent_executor.invoke({"input": user_input})
                    intermediate_steps_str = get_intermediate_steps(out)
                except Exception as e:
                    print(f"Error processing input: {user_input}")
                    print(e)
                    writer.writerow(
                        {'user_input': user_input, 'output': 'Error was made', 'intermediate_steps': 'Error was made'})
                    continue
                # itermediate_steps_summary = summarise_llm_chain.invoke(
                #     {"intermediate_steps_str": intermediate_steps_str})
                writer.writerow(
                    {'user_input': user_input, 'output': out['output'], 'intermediate_steps': intermediate_steps_str})


def concatenate_csv_files(csv_test, csv_result, output_file_path):
    # Read the CSV files
    df_test = pd.read_csv(csv_test)
    df_result = pd.read_csv(csv_result)

    # Merge the DataFrames on 'user_input'
    df_concat = pd.merge(df_test, df_result, on='user_input')

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Save the concatenated DataFrame to the new CSV file
    df_concat.to_csv(output_file_path, index=False)

    print(f"Concatenated CSV saved to {output_file_path}")


EVAL_TEMPLATE = """
I want you to evaluate my agent. Agent's task is to execute the given task by sending commands to a simulator. I will provide you the task and relevant documents that should or are helpful to be used for executing a task. Then I will provide you with agent's intermediate steps and its final output. You must give a score either 0 or 1 as well as a comment for you score. If the task executed without mistakes give a score of 1, if there are mistake or mistakes give a score of 0 and comment why you gave a score of 1 or 0. Sending an unknown command or not valid command is not considered a mistake. Please always check if the command's arguments are in the correct sequence.

TASK: {task}

RELEVANT DOCUMENTS: {relevant_docs}

INTERMEDIATE STEPS: {intermediate_steps}

FINAL ANSWER: {final_answer}

Output your answer in this format:
SCORE: <here goes the score>
COMMENT: <here goes the comment>
"""
eval_prompt = PromptTemplate.from_template(
    EVAL_TEMPLATE
)


evaluator_llm = ChatGroq(
    temperature=0.0, model_name="llama3-70b-8192")

eval_llm_chain = eval_prompt | evaluator_llm | StrOutputParser()


json_schema = {
    "title": "evaluation",
    "description": "information about an evaluation of an agent's performance",
    "type": "object",
    "properties": {
        "score": {"title": "score", "description": "The score from evaluation", "type": "integer"},
        "comment": {"title": "comment", "description": "The comment from evaluation", "type": "string"}
    },
    "required": ["score", "comment"],
}

json_structure_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are extracting information in structured formats."),
        ("human",
         "Use the given format to extract information from the following input: {input}")
    ]
)

structured_json_chain = create_structured_output_chain(
    json_schema, ChatOpenAI(), json_structure_prompt, verbose=False)


def evaluate_csv(input_csv_path, output_csv_path):
    # Read the input CSV file
    df = pd.read_csv(input_csv_path)

    # Ensure the output columns are added
    df['score'] = None
    df['comment'] = None

    # Process each row
    for index, row in df.iterrows():
        # Prepare the input for the LLM chain
        input_data = {
            'task': row['user_input'],
            'relevant_docs': row['docs'],
            'intermediate_steps': row['intermediate_steps'],
            'final_answer': row['output']
        }

        # Invoke the LLM chain and get the result
        result = eval_llm_chain.invoke(input_data)
        print(result)
        result_json = structured_json_chain.run(result)
        print(result_json)
        # Update the dataframe with the results
        df.at[index, 'score'] = result_json['score']
        df.at[index, 'comment'] = result_json['comment']

    # Write the updated dataframe to the output CSV file
    df.to_csv(output_csv_path, index=False)


def create_score_bar_chart(csv_path):
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Extract the score and line number
    df['line_number'] = df.index + 1
    scores = df['score']

    # Adjust the scores for visualization
    # Replace 0 with a small value for visualization
    adjusted_scores = scores.replace(0, 0.01)

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(df['line_number'], adjusted_scores, color=[
            'red' if x == 0.01 else 'green' for x in adjusted_scores])
    plt.xlabel('Task ')
    plt.ylabel('Score')
    plt.title('Score Distribution by Task Number')

    # Calculate the total score
    total_score = scores.sum()

    # Annotate the total score on the figure
    plt.text(0.5, 0.9, f'Total Score: {total_score}', horizontalalignment='center', verticalalignment='center', transform=plt.gca(
    ).transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    # Define the output path
    output_dir = os.path.join(os.path.dirname(csv_path), '..', 'figures')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, os.path.basename(
        csv_path).replace('.csv', '.jpg'))

    # Save the figure
    plt.savefig(output_path, format='jpg')
    plt.close()

    return output_path






process_csv_inputs(csv_input, csv_output, agent_executor)

concatenate_csv_files(csv_input, csv_output, csv_concatinated)

evaluate_csv(csv_concatinated, csv_final)

create_score_bar_chart(csv_final)