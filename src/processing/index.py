import os
import json
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings,
    SimpleDirectoryReader
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class InterviewIndexer:
    def __init__(self, config: dict):
        self.config = config
        self.persist_dir = config.get("persist_dir", "data/processed")
        self.use_openai_embeddings = config.get("use_openai_embeddings", False)
        self.chunk_size = config.get("chunk_size", 512)
        self.chunk_overlap = config.get("chunk_overlap", 50)

        self._init_settings()

    def _init_settings(self):
        load_dotenv()

        model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE")

        Settings.llm = OpenAI(
            model=model_name,
            api_key=api_key,
            api_base=base_url,
            temperature=0.1
        )

        if self.use_openai_embeddings and api_key:
            Settings.embed_model = OpenAIEmbedding(
                model="text-embedding-3-small",
                api_key=api_key,
                api_base=base_url
            )
            logger.info("Using OpenAI embeddings")
        else:
            Settings.embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5",
                trust_remote_code=True
            )
            logger.info("Using HuggingFace embeddings")

    def build_index_from_documents(self, documents: List[dict]) -> Optional[VectorStoreIndex]:
        """Создание индекса из собранных документов"""
        if not documents:
            logger.warning("No documents to index")
            return None

        try:
            # Преобразование в объекты Document
            llama_documents = []
            for doc in documents:
                text = doc.get("text", "")
                metadata = doc.get("metadata", {})

                llama_doc = Document(
                    text=text,
                    metadata=metadata
                )
                llama_documents.append(llama_doc)

            print(f"Creating index from {len(llama_documents)} documents...")

            # Создание индекса
            index = VectorStoreIndex.from_documents(
                llama_documents,
                show_progress=True
            )

            print("Index created successfully")
            return index

        except Exception as e:
            logger.error(f"Error building index: {e}")
            return None

    def build_index_from_directory(self, data_dir: str = "data/raw") -> Optional[VectorStoreIndex]:
        """Создание индекса из локальных файлов (PDF, HTML, JSON)"""
        data_path = Path(data_dir)
        
        if not data_path.exists():
            logger.error(f"Data directory not found: {data_dir}")
            return None
        
        try:
            all_documents = []
            
            # Чтение PDF файлов
            pdf_dir = data_path / "pdf"
            if pdf_dir.exists():
                print(f"Reading PDFs from {pdf_dir}...")
                pdf_reader = SimpleDirectoryReader(
                    input_dir=str(pdf_dir),
                    recursive=False,
                    required_exts=[".pdf"]
                )
                pdf_docs = pdf_reader.load_data()
                all_documents.extend(pdf_docs)
                print(f"  Loaded {len(pdf_docs)} PDF documents")
            
            # Чтение HTML файлов
            html_dir = data_path / "html"
            if html_dir.exists():
                print(f"Reading HTML from {html_dir}...")
                html_reader = SimpleDirectoryReader(
                    input_dir=str(html_dir),
                    recursive=False,
                    required_exts=[".html"]
                )
                html_docs = html_reader.load_data()
                all_documents.extend(html_docs)
                print(f"  Loaded {len(html_docs)} HTML documents")
            
            # Чтение JSON файлов (метаданные статей)
            json_dir = data_path / "json"
            if json_dir.exists():
                print(f"Reading JSON from {json_dir}...")
                for json_file in json_dir.glob("*.json"):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # Извлекаем текст из JSON
                        text = data.get("text", "") or data.get("content", "") or data.get("abstract", "")
                        if text:
                            doc = Document(
                                text=text,
                                metadata={
                                    "source": "json",
                                    "file_name": json_file.name,
                                    "title": data.get("title", json_file.stem)
                                }
                            )
                            all_documents.append(doc)
                    except Exception as e:
                        logger.warning(f"Error reading {json_file}: {e}")
                print(f"  Loaded {len([d for d in all_documents if d.metadata.get('source') == 'json'])} JSON documents")
            
            if not all_documents:
                logger.warning("No documents found in directory")
                return None
            
            print(f"\nTotal documents: {len(all_documents)}")
            print("Creating index...")
            
            # Создание индекса
            index = VectorStoreIndex.from_documents(
                all_documents,
                show_progress=True
            )
            
            # Сохранение индекса
            print(f"Persisting index to {self.persist_dir}...")
            index.storage_context.persist(persist_dir=self.persist_dir)
            
            print("Index created and saved successfully")
            return index
            
        except Exception as e:
            logger.error(f"Error building index from directory: {e}")
            import traceback
            traceback.print_exc()
            return None

    def load_existing_index(self, persist_dir: str = None) -> Optional[VectorStoreIndex]:
        """Загрузка существующего индекса"""
        if not persist_dir:
            persist_dir = self.persist_dir

        if not os.path.exists(persist_dir):
            logger.error(f"Index directory not found: {persist_dir}")
            return None

        try:
            from llama_index.core import StorageContext, load_index_from_storage

            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
            logger.info("Index loaded successfully")
            return index

        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return None


# Функции для обратной совместимости
def init_settings(use_openai_embeddings: bool = True):
    config = {"use_openai_embeddings": use_openai_embeddings}
    indexer = InterviewIndexer(config)
    return indexer


def build_index(data_dir: str = "data/raw", persist_dir: str = "data/processed",
                use_openai_embeddings: bool = False, from_directory: bool = True):
    """
    Построение индекса из локальных файлов или собранных документов.
    
    Args:
        data_dir: Директория с данными
        persist_dir: Директория для сохранения индекса
        use_openai_embeddings: Использовать OpenAI эмбеддинги (иначе HuggingFace)
        from_directory: Читать напрямую из директории (PDF/HTML/JSON)
    """
    config = {
        "persist_dir": persist_dir,
        "use_openai_embeddings": use_openai_embeddings
    }

    indexer = InterviewIndexer(config)
    
    if from_directory:
        # Читаем напрямую из директории
        return indexer.build_index_from_directory(data_dir)
    else:
        # Старый способ - из collected_documents.json
        json_files = list(Path(data_dir).glob("**/collected_documents.json"))
        if not json_files:
            logger.error("No collected documents found")
            return None

        latest_session = max(json_files, key=lambda x: x.stat().st_mtime)

        with open(latest_session, 'r', encoding='utf-8') as f:
            documents = json.load(f)

        return indexer.build_index_from_documents(documents)


def load_existing_index(persist_dir: str = "data/processed",
                        use_openai_embeddings: bool = False):
    config = {
        "persist_dir": persist_dir,
        "use_openai_embeddings": use_openai_embeddings
    }

    indexer = InterviewIndexer(config)
    return indexer.load_existing_index(persist_dir)