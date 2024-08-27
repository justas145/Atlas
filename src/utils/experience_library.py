import os
import chromadb
from dotenv import load_dotenv

load_dotenv()


def initialize_experience_library(collection_name):
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    vectordb_path = os.path.join(base_path, "skills-library", "vectordb")
    
    # Ensure the directory exists
    os.makedirs(vectordb_path, exist_ok=True)
    
    chroma_client = chromadb.PersistentClient(path=vectordb_path)
    openai_ef = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-large",
    )
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def search_experience_library(
    collection, conflict_description, num_ac, conflict_formation
):
    where_full = {
        "$and": [
            {"num_ac": num_ac},
            {"conflict_formation": conflict_formation},
            {"model_name": "gpt_4o_2024_08_06"},
        ]
    }
    where_partial_1 = {
        "$and": [
            {"num_ac": num_ac},
            {"model_name": "gpt_4o_2024_08_06"},
        ]
    }

    search_orders = [
        (where_full, "Full"),
        (where_partial_1, "Partial 1"),
        (None, "No filters"),
    ]

    for where, label in search_orders:
        try:
            query_results = collection.query(
                query_texts=[conflict_description], n_results=1, where=where
            )
            if query_results["documents"] and query_results["documents"][0]:
                doc = (
                    query_results["documents"][0][0]
                    + "\n\n"
                    + query_results["metadatas"][0][0]["commands"]
                )
                return doc
        except Exception as e:
            print(f"Error with {label} query:", e)

    return "No similar conflict found in the database."
