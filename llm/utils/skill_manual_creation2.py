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
from langchain_core.pydantic_v1 import BaseModel, Field

from agent_prompts import (
    do_and_dont_list_prompt,
    conflict_description_prompt,
    dos_donts_list_transformation_prompt,
    relative_values_dos_donts_list,
    final_dos_donts_prompt,
    extraction_prompt,
)

load_dotenv(find_dotenv())

# Adds the directory containing 'prompts' to the Python path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "prompts"))
)
base_path = os.path.dirname(__file__)
vectordb_path = os.path.join(base_path, "skills-library", "vectordb")
chroma_client = chromadb.PersistentClient(path=vectordb_path)
openai_ef = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-large",
)
collection = chroma_client.get_or_create_collection(
    name="skill_manuals",
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"},
)

temperature = 0.4
model_name = "llama3-70b-8192"
llm = ChatGroq(temperature=temperature, model_name=model_name) | StrOutputParser()


class Metadata(BaseModel):
    """Information about a person."""

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
    min_distance: Optional[float] = Field(
        default=None, description="Minimum distance between aircraft in nautical miles"
    )
    min_tcpa: Optional[float] = Field(
        default=None, description="Minimum time to closest point of approach in seconds"
    )

extraction_llm = ChatGroq(temperature=temperature, model_name=model_name)
extraction_runnable = extraction_prompt | extraction_llm.with_structured_output(
    schema=Metadata
)


def extract_commands(data):
    lines = data.split("\n")
    result = []
    capture_next_line = False  # Flag to start capturing next line
    monitoring_active = False  # Flag to handle continued monitoring results
    wait_for_conflict_status = False  # Flag to capture conflict status after delay

    for line in lines:
        line = line.strip()
        if "Invoking:" in line:
            if "SENDCOMMAND" in line or "GETCONFLICTINFO" in line:
                result.append(line)
                capture_next_line = True  # Start capturing next relevant output
            elif "CONTINUEMONITORING" in line:
                result.append(line)
                monitoring_active = True  # Start monitoring section
            else:
                capture_next_line = False  # Ensure no capture unless specified
                monitoring_active = False  # Stop monitoring unless specified
                wait_for_conflict_status = False  # Reset waiting status
        elif capture_next_line:
            if "Command executed successfully." in line:
                result.append(line)
                capture_next_line = False  # Command feedback captured, stop capturing
            elif (
                "Number of aircraft in conflict:" in line
                or "No conflicts detected" in line
            ):
                result.append(line)
                capture_next_line = False  # Stop capturing unless it's monitoring
        elif monitoring_active:
            if "After" in line:
                result.append(line)
                wait_for_conflict_status = (
                    True  # After stating delay, wait for conflict status
                )
            elif wait_for_conflict_status and (
                "Number of aircraft in conflict:" in line
                or "No conflicts detected" in line
            ):
                result.append(line)
                monitoring_active = False  # End monitoring on conflict info
                wait_for_conflict_status = False  # Reset waiting status
        elif "Crash Alert:" in line:
            result.append(line)  # Capture crash alerts as they appear

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
                result.extend(section_data)  # Append collected data to the result
                sections_captured.add(current_section)  # Mark the section as captured
                section_data = []  # Reset for the next section

            if (
                any(
                    keyword in line
                    for keyword in ["GETALLAIRCRAFTINFO", "GETCONFLICTINFO"]
                )
                and line not in sections_captured
            ):
                current_section = line  # Start a new section
            else:
                current_section = None  # Not a section we're collecting

        if current_section:
            section_data.append(line)  # Collect data for the current section

    # Append the last section if it wasn't already added
    if section_data and current_section not in sections_captured:
        result.extend(section_data)

    return "\n".join(result)


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

def create_experience_doc(console_output):
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
        relative_values_dos_donts_list.format(dos_donts_list=dos_donts_list_transformation)
    )
    final_dos_donts_list = llm.invoke(
        final_dos_donts_prompt.format(
            conflict_description=conflict_description,
            commands_list=relative_values_dos_donts_list,
        )
    )

    experience_doc = conflict_description + "\n\n" + final_dos_donts_list
    metadata = extraction_runnable.invoke({"text": conflict_description})
    return experience_doc, metadata


def update_skill_library(collection, skill_manual, metadata):
    # get documents by metadata filter
    # where = {
    #     "$and": [
    #         {"num_ac": metadata.num_ac},
    #         {"conflict_type": metadata.conflict_type},
    #         {"conflict_formation": metadata.conflict_formation},
    #         {"num_commands": metadata.num_commands}
    #     ]
    # }

    uuid4 = uuid.uuid4()
    try:
        collection.upsert(
            ids=[str(uuid4)],
            documents=[skill_manual],
            metadatas=[
                {
                    "num_ac": metadata.num_ac,
                    "conflict_formation": metadata.conflict_formation,
                    "min_distance": metadata.min_distance,
                    "min_tcpa": metadata.min_tcpa,
                }
            ],
        )
        print("Skill manual added to the collection")
    except Exception as e:
        print(f"Error adding skill manual to the collection: {e}")
