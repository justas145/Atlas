import re
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import os
import sys
import chromadb
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import uuid
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
# Adds the directory containing 'prompts' to the Python path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "prompts"))
)
from agent_prompts import (
    do_and_dont_list_prompt,
    conflict_description_prompt,
    dos_donts_list_transformation_prompt,
    relative_values_dos_donts_list_prompt,
    final_dos_donts_prompt,
    extraction_metada_prompt,
    # anonymous_values_dos_donts_list_prompt,
)

load_dotenv(find_dotenv())


# base_path = os.path.dirname(__file__)
base_path = "C:\\Users\\justa\\OneDrive\\Desktop\\Developer\\LLM-Enhanced-ATM\\llm"
print(base_path)
vectordb_path = os.path.join(base_path, "skills-library", "vectordb")
chroma_client = chromadb.PersistentClient(path=vectordb_path)
openai_ef = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-large",
)
#chroma_client.delete_collection("experience_library_v3")
collection = chroma_client.get_or_create_collection(
    name="experience_library_v3",
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"},
)


def key_generator():
    # Load keys from environment variables
    key1 = os.getenv("GROQ_API_KEY_1")
    key2 = os.getenv("GROQ_API_KEY_2")
    key3 = os.getenv("GROQ_API_KEY_3")
    while True:
        yield key1
        yield key2
        yield key3


# Create an instance of the generator
key_cycle = key_generator()


# Function to get the next key
def get_next_key():
    return next(key_cycle)


class Metadata(BaseModel):
    """Information about an air traffic conflict."""

    # This doc-string is sent to the LLM as the description of the schema Metadata,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    num_ac: Optional[int] = Field(
        default=None, description="Number of aircraft in conflict"
    )
    conflict_formation: Optional[str] = Field(
        default=None, description="conflict formation"
    )


def extract_commands(data):
    lines = data.split("\n")
    result = []
    capture_flag = False  # Flag to start capturing after certain keywords

    for line in lines:
        line = line.strip()
        if "Invoking:" in line and "`GETALLAIRCRAFTINFO`" not in line:
            result.append(line)
            capture_flag = True  # Start capturing the results after invoking
        elif capture_flag:
            if (
                "Number of aircraft pairs in conflict:" in line
                or "No conflicts detected" in line
            ):
                result.append(line)
                capture_flag = False  # Stop capturing after conflict info
            elif "Command executed successfully." in line:
                result.append(line)
                capture_flag = False  # Stop capturing after command execution
            elif "Crash Alert:" in line:
                result.append(line)
                capture_flag = False  # Include crash alerts and stop capturing

    return "\n".join(result)


def filter_first_instance_only(text):
    keywords = ["GETALLAIRCRAFTINFO", "GETCONFLICTINFO", "CONTINUEMONITORING"]
    lines = text.split("\n")
    result = []
    captured = set()
    capture = False

    for line in lines:
        if any(f"Invoking: `{keyword}`" in line for keyword in keywords):
            section_keyword = next(
                keyword for keyword in keywords if f"Invoking: `{keyword}`" in line
            )
            if section_keyword not in captured:
                capture = True
                captured.add(section_keyword)
                if (
                    result
                ):  # Add a new line before starting a new section if the result is not empty
                    result.append("")
            else:
                capture = False  # Stop capturing lines if this section starts again
        if capture:
            result.append(line)  # Append the line to the result if we are capturing

    return "\n".join(result)


def extract_conflict_info(data):
    lines = data.split("\n")
    result = []
    section_data = []
    current_section = None
    sections_captured = set()

    for line in lines:
        line = line.strip()
        if "Invoking:" in line:
            if section_data:  # There's collected data from the previous section
                if current_section not in sections_captured:
                    result.extend(section_data)  # Append collected data to the result
                    sections_captured.add(
                        current_section
                    )  # Mark the section as captured
                section_data = []  # Reset for the next section

            if any(
                keyword in line
                for keyword in [
                    "GETALLAIRCRAFTINFO",
                    "GETCONFLICTINFO",
                    "CONTINUEMONITORING",
                ]
            ):
                current_section = line  # Start a new section
            else:
                current_section = None  # Not a section we're collecting

        if current_section:
            section_data.append(line)  # Collect data for the current section

    # Append the last section if it wasn't already added
    if section_data and current_section not in sections_captured:
        result.extend(section_data)

    output = "\n".join(result)
    clean_output = filter_first_instance_only(output)

    return clean_output


def extract_aircraft_init_info(text):
    lines = text.split("\n")
    collecting = False
    result = []

    for line in lines:
        if "Aircraft idx:" in line:
            collecting = True  # Start collecting lines
        if "Invoking:" in line and collecting:
            break  # Stop collecting lines when encountering this keyword
        if collecting:
            result.append(line)  # Collect lines if we are in the collecting state

    return "\n".join(result)


def replace_flight_names(text):
    # Create a pattern that matches "flight" followed by a number, case insensitive
    pattern = re.compile(r"flight(\d+)", re.IGNORECASE)

    # Define a function to determine the replacement based on the captured number
    def replace(match):
        number = int(match.group(1))  # The number following "flight"
        return f"AIRCRAFT_{chr(64 + number)}"  # Convert number to letter (1 -> A, 2 -> B, etc.)

    # Use the sub method to replace all occurrences using the replace function
    result = pattern.sub(replace, text)
    return result


def create_experience_doc(console_output, model_name="llama3-70b-8192", temperature=0.3):
    model_name = model_name.replace('_', '-')
    if 'gpt' in model_name:
        print("Using OpenAI model")
        llm = ChatOpenAI(temperature=temperature, model_name=model_name) | StrOutputParser()
        extraction_llm= ChatOpenAI(temperature=temperature, model_name=model_name)

    else:
        print("Using Groq model")
        api_key = get_next_key()  # Get the next API key for this call
        llm = ChatGroq(temperature=temperature, model_name=model_name, api_key=api_key) | StrOutputParser()
        extraction_llm = ChatGroq(temperature=temperature, model_name=model_name, api_key=api_key)
    extraction_runnable = (
        extraction_metada_prompt
        | extraction_llm.with_structured_output(schema=Metadata)
    )
    # Process the input data
    processed_log_commands = extract_commands(console_output)
    processed_log_conflict_info = extract_conflict_info(console_output)
    aircraft_init_info = extract_aircraft_init_info(processed_log_conflict_info)
    # create conflict description
    conflict_description = llm.invoke(
        conflict_description_prompt.format(log=processed_log_conflict_info)
    )

    # create a do's and don'ts list
    dos_donts_list = llm.invoke(do_and_dont_list_prompt.format(log=processed_log_commands))

    # transform the do's and don'ts list by adding aircraft state before and after the commands
    dos_donts_list_transformation = llm.invoke(
        dos_donts_list_transformation_prompt.format(
            init_aircraft_info=aircraft_init_info, dos_donts_list=dos_donts_list
        )
    )

    # create a relative values do's and don'ts list. For example instead of ALT A1 10000 is a do, it should be transformed to increase altitude of A1 by ... ft
    relative_values_dos_donts_list = llm.invoke(
        relative_values_dos_donts_list_prompt.format(dos_donts_list=dos_donts_list_transformation)
    )

    final_dos_donts_list = llm.invoke(
        final_dos_donts_prompt.format(
            conflict_description=conflict_description,
            commands_list=relative_values_dos_donts_list,
        )
    )

    experience_doc = conflict_description + "\n\n" + final_dos_donts_list
    experience_doc = replace_flight_names(experience_doc)

    conflict_description = replace_flight_names(conflict_description)
    final_dos_donts_list = replace_flight_names(final_dos_donts_list)

    # print(experience_doc)
    metadata = extraction_runnable.invoke({"text": conflict_description})
    return experience_doc, metadata, conflict_description, final_dos_donts_list


def update_experience_library(
    collection, conflict_description, final_dos_donts_list, metadata, model_name
):

    uuid4 = uuid.uuid4()

    try:
        collection.upsert(
            ids=[str(uuid4)],
            documents=[conflict_description],
            metadatas=[
                {
                    "num_ac": metadata.num_ac,
                    "conflict_formation": metadata.conflict_formation,
                    "model_name": model_name,
                    "commands": final_dos_donts_list,
                }
            ],
        )
        print("Skill manual added to the collection")
    except Exception as e:
        print(f"Error adding skill manual to the collection: {e}")


import pandas as pd

csv_path = "../results/main/sa_ma_no_exp_V3.csv"
data = pd.read_csv(csv_path)

# Define the maximum number of attempts
max_attempts = 5

# Iterate over each row in the DataFrame
for index, row in data.iterrows():
    if row["num_send_commands"] > 7:
        print(f"Skipping row {index} as num_send_commands is greater than 7.")
        continue
    if 'multi' in row["agent_type"].lower().strip():
        print(f"Skipping row {index} as agent_type is multi.")
        continue

    print(f"Processing row {index}")
    print(row["agent_type"])
    console_output = row["log"]
    model_name = row["model_name"]

    # Attempt to create the experience document and metadata up to `max_attempts` times
    for attempt in range(max_attempts):
        try:
            experience_doc, metadata, conflict_description, final_dos_donts_list = (
                create_experience_doc(console_output, model_name)
            )

            # Check if 'please' is in the experience document
            if "please" in experience_doc.lower():
                print("Generated Experience Document contains 'please':")
                print(experience_doc)

                # Ask for user input to decide whether to keep the experience document
                user_input = (
                    input("Do you want to keep this document? [y/n]: ").strip().lower()
                )
                if user_input == "n":
                    print(f"Skipping row {index} as per user input.")
                    break  # Move to the next data row

            # If successful and user decided to keep, update the experience library
            update_experience_library(
                collection,
                conflict_description,
                final_dos_donts_list,
                metadata,
                model_name,
            )
            print(f"Successfully updated library for row {index}")
            break  # Break the retry loop after successful update
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for row {index}, error: {e}")
            if attempt == max_attempts - 1:
                # If the last attempt fails, log and move on
                print(f"Skipping row {index} after {max_attempts} failed attempts.")
