import sys
sys.path.append('../bluesky')  # Adjust the path as necessary

from bluesky.network.client import Client
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv, find_dotenv
import time
from contextlib import contextmanager
from io import StringIO
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.agents import tool
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
import bluesky
import streamlit as st
import os


# Load the .env file
load_dotenv(find_dotenv())

# blue sky client

client = Client()
client.connect("127.0.0.1", 11000, 11001)
client.update()

client.update()

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-large"
)
vectordb_path = 'skills-library/vectordb'
chroma_client = chromadb.PersistentClient(path=vectordb_path)
# Get a collection object from an existing collection, by name. If it doesn't exist, create it.
collection = chroma_client.get_or_create_collection(
    name="test1", embedding_function=openai_ef, metadata={"hnsw:space": "cosine"})


# initialize the GPT-3.5 model
output_parser = StrOutputParser()
llm_input_restructure = ChatOpenAI()
chain_input_restructure = llm_input_restructure | output_parser


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


def write_colored(text, color):
    # Using markdown with unsafe_allow_html to render HTML with custom styles
    st.markdown(
        f'<span style="color:{color}">{text}</span>', unsafe_allow_html=True)


def display_actions_and_results(out):
    actions = out["intermediate_steps"]
    final_output = out["output"]
    for action in actions:
        if isinstance(action, tuple) and len(action) == 2:
            tool_action, results = action
            # Display the invoking part in blue
            write_colored(
                f"Invoking: `{tool_action.tool}` with `{tool_action.tool_input}`", "blue")
            if isinstance(results, list):
                # For tavily_search_results_json action in green
                if tool_action.tool == 'tavily_search_results_json':
                    for result in results:
                        # Replace newlines with HTML line breaks
                        content = result['content'].replace('\n', '<br>')
                        write_colored(
                            f"[{{'url': '{result['url']}', 'content': '{content}'}}]", "green")
                else:
                    # For send_command_to_simulator actions and others in cyan
                    write_colored(" ".join(results), "cyan")
            elif isinstance(results, str):
                # Direct string results (e.g., syntax errors or simple responses) in red
                write_colored(results, "red")
            st.write("---")  # Visual separator in Streamlit
    write_colored(final_output, "purple")

@tool
def get_all_aircraft_info(command: str) -> str:
    """Returns each aircraft information. Use command GETACIDS to get all aircrafts."""
    client.send_event(b'STACK', command)
    sim_output = update_until_complete(client)
    return sim_output

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

# tool to query chroma and extract relevant results. output is a list
@tool
def query_database(input: str) -> list:
    """Queries the vector database with the given input and returns top 5 results."""
    query_results = collection.query(
        query_texts=[input],
        n_results=5
    )
    return query_results['documents'][0]

tavily_tool = TavilySearchResults()

tools = [send_command_to_simulator,
         get_all_aircraft_info, query_database, tavily_tool]

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
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm_with_tools = llm.bind_tools(tools)
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
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)




# open basecmds as a string
with open('prompts/basecmds.txt', 'r') as file:
    base_cmds = file.read()


prompt_template = PromptTemplate.from_template(
    "{input} \n Below is some documentation that can either be useful or not for you to use in order to generate and send a correct command to the simulator. \n\n {doc1} \n {doc2} \n {doc3}."
)



def main():
    st.title("LLM Enhances ATM")

    # Step 2: User input
    user_input = st.text_input("Enter your input:")

    if user_input:
        # # Step 3: Show input_restructured
        # input_restructured = chain_input_restructure.invoke(
        #     f"{user_input} \n Here are the commands available to use: \n {base_cmds} \n\n Based on these commands, how should I proceed?"
        # )
        # st.subheader("Input Restructured")
        # st.write(input_restructured)

        # Step 4: Show extracted documents from Chroma
        query_results = collection.query(
            query_texts=[user_input],
            n_results=5
        )
        st.subheader("Extracted Documents")
        for i, doc in enumerate(query_results['documents'][0]):
            st.write(f"Document {i+1}: {doc}")

        # Step 5: Show final input
        final_input = prompt_template.format(
            input=user_input,
            doc1=query_results['documents'][0][0],
            doc2=query_results['documents'][0][1],
            doc3=query_results['documents'][0][2]
        )
        st.subheader("Final Input")
        st.write(final_input)

        # Step 6: Show LLM agent's thinking
        st.subheader("LLM Agent's Thinking")
        out = agent_executor({"input": final_input})
        display_actions_and_results(out)

if __name__ == "__main__":
    main()
