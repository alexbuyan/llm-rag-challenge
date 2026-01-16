"""
Conversational RAG Chat Service for the Interview Preparation System.

Provides a chat interface that maintains conversation history and uses
hybrid retrieval (vector + BM25) for finding relevant study materials.
"""

import os
import re
from typing import Dict, Any, List, Optional, Generator
from dataclasses import dataclass

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def strip_html_tags(text: str) -> str:
    """Remove HTML tags and clean up whitespace."""
    # Remove HTML tags
    clean = re.sub(r'<[^>]+>', ' ', text)
    # Remove extra whitespace
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean


def extract_title_from_filename(filename: str) -> str:
    """Extract a readable title from a filename."""
    if not filename:
        return "Unknown"
    # Remove extension and clean up
    name = re.sub(r'\.[^.]+$', '', filename)
    # Remove leading numbers/IDs
    name = re.sub(r'^\d+\s*', '', name)
    return name if name else "Unknown"

from llama_index.core import Settings
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv

from src.processing.index import load_existing_index
from src.processing.rag import HybridRetriever

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ChatResponse:
    """Represents a chat response with sources."""
    message: str
    sources: List[Dict[str, Any]]
    
    def __str__(self) -> str:
        return self.message


class ChatService:
    """
    Conversational RAG service that maintains chat history and provides
    context-aware responses using hybrid retrieval.
    """
    
    def __init__(
        self,
        persist_dir: str = "data/processed",
        llm_model: str = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        similarity_top_k: int = 5,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
        memory_token_limit: int = 3000
    ):
        self.persist_dir = persist_dir
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.similarity_top_k = similarity_top_k
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.memory_token_limit = memory_token_limit
        
        self.index = None
        self.chat_engine = None
        self.hybrid_retriever = None
        self.memory = None
        self._last_sources: List[Dict[str, Any]] = []
        
        self._initialized = False
    
    def initialize(self) -> bool:
        """
        Initialize the chat service by loading the index and setting up components.
        Returns True if successful, False otherwise.
        """
        if self._initialized:
            return True
            
        try:
            load_dotenv()

            # Setup LLM (Mistral AI)
            model_name = self.llm_model or os.getenv("MISTRAL_MODEL_NAME", "mistral-large-latest")
            api_key = os.getenv("MISTRAL_API_KEY")
            
            if not api_key:
                logger.error("MISTRAL_API_KEY not found in environment")
                return False

            logger.info(f"Initializing MistralAI with: model={model_name}, temperature={self.temperature}, max_tokens={self.max_tokens}")
            
            Settings.llm = MistralAI(
                model=model_name,
                api_key=api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                additional_kwargs={"max_tokens": self.max_tokens}
            )
            
            # Setup embeddings
            Settings.embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5",
                trust_remote_code=True,
                device="cpu"
            )
            
            # Load the existing index
            logger.info(f"Loading index from {self.persist_dir}...")
            self.index = load_existing_index(
                persist_dir=self.persist_dir
            )
            
            if not self.index:
                logger.error("Failed to load index")
                return False
            
            # Setup hybrid retriever
            retriever_config = {
                "similarity_top_k": self.similarity_top_k,
                "bm25_weight": self.bm25_weight,
                "vector_weight": self.vector_weight
            }
            self.hybrid_retriever = HybridRetriever(self.index, retriever_config)
            
            # Setup chat memory
            self.memory = ChatMemoryBuffer.from_defaults(
                token_limit=self.memory_token_limit
            )
            
            # Create chat engine with context
            # Custom context prompt to encourage comprehensive answers
            context_prompt = (
                "Here is relevant context from the knowledge base:\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Using the above context, provide a detailed and comprehensive answer to the question. "
                "Do not cut your response short - provide a complete answer."
            )
            
            self.chat_engine = CondensePlusContextChatEngine.from_defaults(
                retriever=self.index.as_retriever(similarity_top_k=self.similarity_top_k),
                memory=self.memory,
                llm=Settings.llm,
                system_prompt=self._get_system_prompt(),
                context_prompt=context_prompt,
                skip_condense=True,
                verbose=False
            )
            
            self._initialized = True
            logger.info("Chat service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing chat service: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _get_system_prompt(self) -> str:
        """Return the system prompt for the chat engine."""
        return """You are an expert interview preparation assistant. Your role is to help users 
prepare for technical interviews by providing clear, accurate, and practical information.

When answering questions:
1. Be concise but comprehensive
2. Use examples when helpful
3. Highlight key concepts and best practices
4. If discussing algorithms or system design, mention time/space complexity when relevant
5. Provide actionable study tips when appropriate

Base your answers on the provided context from the knowledge base. If the context doesn't 
contain relevant information, say so clearly and provide general guidance based on your knowledge.

Format your responses using markdown for better readability:
- Use **bold** for key terms
- Use bullet points for lists
- Use code blocks for code examples
- Use headers for organizing longer responses"""

    def chat(self, message: str, use_hybrid: bool = True) -> ChatResponse:
        """
        Send a message and get a response.
        
        Args:
            message: The user's message
            use_hybrid: Whether to use hybrid retrieval for sources
            
        Returns:
            ChatResponse with message and sources
        """
        if not self._initialized:
            if not self.initialize():
                return ChatResponse(
                    message="Error: Failed to initialize chat service. Please check your configuration.",
                    sources=[]
                )
        
        try:
            # Get sources using hybrid retrieval
            self._last_sources = []
            if use_hybrid and self.hybrid_retriever:
                results = self.hybrid_retriever.retrieve(message)
                self._last_sources = self._format_sources(results)
            
            # Get chat response
            response = self.chat_engine.chat(message)
            
            return ChatResponse(
                message=str(response),
                sources=self._last_sources
            )
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return ChatResponse(
                message=f"Error processing your message: {str(e)}",
                sources=[]
            )
    
    def stream_chat(self, message: str, use_hybrid: bool = True) -> Generator[str, None, ChatResponse]:
        """
        Stream a chat response token by token.
        
        Args:
            message: The user's message
            use_hybrid: Whether to use hybrid retrieval for sources
            
        Yields:
            Response tokens as they become available
            
        Returns:
            Final ChatResponse with complete message and sources
        """
        if not self._initialized:
            if not self.initialize():
                yield "Error: Failed to initialize chat service."
                return ChatResponse(
                    message="Error: Failed to initialize chat service.",
                    sources=[]
                )
        
        try:
            # Get sources using hybrid retrieval
            self._last_sources = []
            if use_hybrid and self.hybrid_retriever:
                results = self.hybrid_retriever.retrieve(message)
                self._last_sources = self._format_sources(results)
            
            # Stream chat response
            streaming_response = self.chat_engine.stream_chat(message)
            
            full_response = ""
            token_count = 0
            for token in streaming_response.response_gen:
                full_response += token
                token_count += 1
                yield token
            
            logger.info(f"Stream completed: {token_count} tokens, {len(full_response)} chars")
            
            return ChatResponse(
                message=full_response,
                sources=self._last_sources
            )
            
        except Exception as e:
            logger.error(f"Error in stream_chat: {e}")
            error_msg = f"Error processing your message: {str(e)}"
            yield error_msg
            return ChatResponse(message=error_msg, sources=[])
    
    def _format_sources(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format retrieval results as source citations."""
        sources = []
        for i, result in enumerate(results, 1):
            node = result.get("node")
            if not node:
                continue
                
            metadata = node.node.metadata
            content = result.get("content", "")
            
            # Clean HTML from content
            clean_content = strip_html_tags(content)
            
            # Get title - fallback to extracting from filename if unknown
            title = metadata.get("title", "")
            if not title or title == "Unknown":
                title = extract_title_from_filename(metadata.get("file_name", ""))
            
            # Determine source type from file extension
            file_name = metadata.get("file_name", "")
            source_type = metadata.get("source", "")
            if not source_type or source_type == "Unknown":
                if file_name.endswith('.pdf'):
                    source_type = "PDF"
                elif file_name.endswith('.html'):
                    source_type = "HTML"
                elif file_name.endswith('.json'):
                    source_type = "JSON"
                else:
                    source_type = "Document"
            
            source = {
                "id": i,
                "title": title,
                "source": source_type,
                "url": metadata.get("url", ""),
                "file_name": file_name,
                "combined_score": round(result.get("combined_score", 0), 3),
                "vector_score": round(result.get("vector_score", 0), 3),
                "bm25_score": round(result.get("bm25_score", 0), 3),
                "content_preview": clean_content[:300] + "..." if len(clean_content) > 300 else clean_content
            }
            sources.append(source)
        
        return sources
    
    def get_last_sources(self) -> List[Dict[str, Any]]:
        """Get the sources from the last query."""
        return self._last_sources
    
    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        if self.memory:
            self.memory.reset()
        self._last_sources = []
        logger.info("Conversation history reset")
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get the current chat history."""
        if not self.memory:
            return []
        
        history = []
        for msg in self.memory.get_all():
            history.append({
                "role": msg.role.value,
                "content": msg.content
            })
        return history
    
    @property
    def is_initialized(self) -> bool:
        """Check if the service is initialized."""
        return self._initialized

