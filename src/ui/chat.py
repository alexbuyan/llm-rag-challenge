"""
Conversational RAG Chat Service for the Interview Preparation System.

Provides a chat interface that maintains conversation history and uses
hybrid retrieval (vector + BM25) for finding relevant study materials.
"""

import os
from typing import Dict, Any, List, Optional, Generator
from dataclasses import dataclass

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from llama_index.core import Settings
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI
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
        use_openai_embeddings: bool = False,
        llm_model: str = None,
        temperature: float = 0.1,
        similarity_top_k: int = 5,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
        memory_token_limit: int = 3000
    ):
        self.persist_dir = persist_dir
        self.use_openai_embeddings = use_openai_embeddings
        self.llm_model = llm_model
        self.temperature = temperature
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
            
            # Setup LLM
            model_name = self.llm_model or os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_API_BASE")
            
            if not api_key:
                logger.error("OPENAI_API_KEY not found in environment")
                return False
            
            Settings.llm = OpenAI(
                model=model_name,
                api_key=api_key,
                api_base=base_url,
                temperature=self.temperature
            )
            
            # Setup embeddings
            Settings.embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5",
                trust_remote_code=True
            )
            
            # Load the existing index
            logger.info(f"Loading index from {self.persist_dir}...")
            self.index = load_existing_index(
                persist_dir=self.persist_dir,
                use_openai_embeddings=self.use_openai_embeddings
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
            self.chat_engine = CondensePlusContextChatEngine.from_defaults(
                retriever=self.index.as_retriever(similarity_top_k=self.similarity_top_k),
                memory=self.memory,
                llm=Settings.llm,
                system_prompt=self._get_system_prompt(),
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
            for token in streaming_response.response_gen:
                full_response += token
                yield token
            
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
            
            source = {
                "id": i,
                "title": metadata.get("title", "Unknown"),
                "source": metadata.get("source", "Unknown"),
                "url": metadata.get("url", ""),
                "file_name": metadata.get("file_name", ""),
                "combined_score": round(result.get("combined_score", 0), 3),
                "vector_score": round(result.get("vector_score", 0), 3),
                "bm25_score": round(result.get("bm25_score", 0), 3),
                "content_preview": content[:300] + "..." if len(content) > 300 else content
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

