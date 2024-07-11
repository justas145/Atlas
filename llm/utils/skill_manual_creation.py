import os
import sys
from io import StringIO
import chromadb
import dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import uuid

from typing import Optional

from langchain_core.pydantic_v1 import BaseModel, Field


# Adds the directory containing 'prompts' to the Python path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "prompts"))
)

from agent_prompts import conflict_description_prompt, conflict_solution_prompt

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

collection = chroma_client.get_or_create_collection(
    name="skill_manuals",
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"},
)
llm = ChatGroq(temperature=temperature, model_name=model_name) | StrOutputParser()


class Metadata(BaseModel):
    """Information about a person."""

    # This doc-string is sent to the LLM as the description of the schema Metadata,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    num_ac: Optional[str] = Field(default=None, description="Number of aircraft in conflict")
    conflict_formation: Optional[str] = Field(
        default=None, description="conflict formation"
    )
    num_commands: Optional[int] = Field(default=None, description="Number of commands sent")

extraction_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        # Please see the how-to about improving performance with
        # reference examples.
        # MessagesPlaceholder('examples'),
        ("human", "{text}"),
    ]
)
extraction_llm = ChatGroq(temperature=temperature, model_name=model_name)
extraction_runnable = extraction_prompt | extraction_llm.with_structured_output(schema=Metadata)


def create_skill_manual(log):
    conflict_description_template = conflict_description_prompt.format(log=log)
    conflict_solution_template = conflict_solution_prompt.format(log=log)

    conflict_description = llm.invoke(conflict_description_template)
    conflict_solution = llm.invoke(conflict_solution_template)
    metadata = extraction_runnable.invoke({"text": conflict_description})
    # metadata is a class instance of Metadata with attributes num_ac, conflict_type, and conflict_formation
    skill_manual = conflict_description + "\n" + conflict_solution
    return skill_manual, metadata



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
            metadatas=[{"num_ac": metadata.num_ac, "conflict_formation": metadata.conflict_formation, "num_commands": metadata.num_commands}]
        )
        print("Skill manual added to the collection")
    except Exception as e:
        print(f"Error adding skill manual to the collection: {e}")
