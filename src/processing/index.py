import os
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def init_settings():
    """Initialize global settings with environment variables."""
    load_dotenv()
    
    model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")

    # Configure LLM
    Settings.llm = OpenAI(
        model=model_name,
        api_key=api_key,
        api_base=base_url,
    )
    
    # Configure Embeddings (Local HuggingFace)
    # using bge-small-en-v1.5 for a good balance of speed and performance
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

def build_index(data_dir: str = "data/raw", persist_dir: str = "data/processed"):
    """
    Builds a vector index from documents in data_dir and persists it.
    """
    init_settings()
    
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' not found.")
        return None

    print("Loading documents...")
    documents = SimpleDirectoryReader(data_dir, recursive=True).load_data()
    print(f"Loaded {len(documents)} documents.")

    print("Building index...")
    index = VectorStoreIndex.from_documents(documents)
    
    print(f"Persisting index to '{persist_dir}'...")
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)
        
    index.storage_context.persist(persist_dir=persist_dir)
    print("Index built and saved.")
    return index

def load_existing_index(persist_dir: str = "data/processed"):
    """
    Loads an existing index from storage.
    """
    init_settings()
    
    if not os.path.exists(persist_dir):
        print(f"Persistence directory '{persist_dir}' not found. Please build index first.")
        return None

    print(f"Loading index from '{persist_dir}'...")
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    return index

if __name__ == "__main__":
    # Example usage
    build_index()

