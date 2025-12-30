"""
Streamlit Chat Application for Interview Preparation RAG System.

Run with: uv run streamlit run src/ui/app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ui.chat import ChatService


# Page configuration
st.set_page_config(
    page_title="Interview Prep Assistant",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background-color: transparent;
    }
    
    /* Source card styling */
    .source-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #e94560;
    }
    
    .source-title {
        color: #e94560;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .source-meta {
        color: #a0a0a0;
        font-size: 0.85rem;
    }
    
    .score-badge {
        display: inline-block;
        background: #e94560;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin-right: 5px;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 1rem 0 2rem 0;
    }
    
    .main-header h1 {
        background: linear-gradient(120deg, #e94560, #0f3460);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #e94560 0%, #0f3460 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(233, 69, 96, 0.4);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_service" not in st.session_state:
        st.session_state.chat_service = None
    
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    
    if "init_error" not in st.session_state:
        st.session_state.init_error = None


def initialize_chat_service():
    """Initialize the chat service if not already done."""
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Check for API key first
    if not os.getenv("OPENAI_API_KEY"):
        st.session_state.init_error = "OPENAI_API_KEY not found. Please create a .env file with your API key (see env.example)."
        return False
    
    if st.session_state.chat_service is None:
        st.session_state.chat_service = ChatService(
            persist_dir="data/processed",
            use_openai_embeddings=False,
            similarity_top_k=5,
            bm25_weight=0.3,
            vector_weight=0.7
        )
    
    if not st.session_state.initialized:
        with st.spinner("Loading knowledge base..."):
            success = st.session_state.chat_service.initialize()
            if success:
                st.session_state.initialized = True
                st.session_state.init_error = None
            else:
                st.session_state.init_error = "Failed to initialize. Check that data/processed/ contains a valid index."
                return False
    
    return True


def render_sidebar():
    """Render the sidebar with system info and controls."""
    with st.sidebar:
        st.markdown("## 🎯 Interview Prep Assistant")
        st.markdown("---")
        
        # System status
        st.markdown("### System Status")
        if st.session_state.initialized:
            st.success("✅ Knowledge base loaded")
        elif st.session_state.init_error:
            st.error(f"❌ {st.session_state.init_error}")
        else:
            st.warning("⏳ Initializing...")
        
        st.markdown("---")
        
        # Conversation controls
        st.markdown("### Conversation")
        
        if st.button("🔄 New Conversation", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.chat_service:
                st.session_state.chat_service.reset_conversation()
            st.rerun()
        
        # Message count
        msg_count = len(st.session_state.messages)
        st.caption(f"Messages: {msg_count}")
        
        st.markdown("---")
        
        # About section
        st.markdown("### About")
        st.markdown("""
        This assistant helps you prepare for technical interviews by providing:
        
        - 📚 Study materials from research papers
        - 💡 Key concepts and explanations  
        - 🎯 Practical interview tips
        - 🔍 Relevant source citations
        """)
        
        st.markdown("---")
        
        # Tips
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Ask about specific topics like "system design" or "algorithms"
        - Request explanations of complex concepts
        - Ask follow-up questions for deeper understanding
        - Check the sources for additional reading
        """)


def render_sources(sources):
    """Render source citations in an expander."""
    if not sources:
        return
    
    with st.expander(f"📚 View Sources ({len(sources)} references)", expanded=False):
        for source in sources:
            st.markdown(f"""
            <div class="source-card">
                <div class="source-title">📄 {source['title']}</div>
                <div class="source-meta">
                    <span class="score-badge">Score: {source['combined_score']}</span>
                    <span>Source: {source['source']}</span>
                    {f" | File: {source['file_name']}" if source['file_name'] else ""}
                </div>
                <div style="margin-top: 0.5rem; color: #d0d0d0; font-size: 0.9rem;">
                    {source['content_preview']}
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_chat_interface():
    """Render the main chat interface."""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Interview Preparation Assistant</h1>
        <p style="color: #a0a0a0;">Ask questions about technical interviews, algorithms, system design, and more</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show setup instructions if there's an initialization error
    if st.session_state.init_error and "OPENAI_API_KEY" in st.session_state.init_error:
        st.warning("### ⚠️ Setup Required")
        st.markdown("""
        To use this application, you need to configure your OpenAI API key:
        
        1. Create a `.env` file in the project root (copy from `env.example`)
        2. Add your OpenAI API key: `OPENAI_API_KEY=sk-your-key-here`
        3. Restart the application
        
        ```bash
        cp env.example .env
        # Edit .env and add your API key
        ```
        """)
        return
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                render_sources(message["sources"])
    
    # Chat input
    if prompt := st.chat_input("Ask about interview topics...", key="chat_input"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            if not initialize_chat_service():
                st.error("Chat service not available. Please check your setup.")
                return
            
            # Stream the response
            response_placeholder = st.empty()
            full_response = ""
            sources = []
            
            try:
                # Use streaming for better UX
                stream_gen = st.session_state.chat_service.stream_chat(prompt)
                
                for token in stream_gen:
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")
                
                # Remove cursor and show final response
                response_placeholder.markdown(full_response)
                
                # Get sources after streaming completes
                sources = st.session_state.chat_service.get_last_sources()
                
            except Exception as e:
                full_response = f"Error: {str(e)}"
                response_placeholder.error(full_response)
            
            # Show sources
            render_sources(sources)
            
            # Add assistant message to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources
            })


def main():
    """Main application entry point."""
    initialize_session_state()
    
    # Eagerly initialize chat service on page load
    if not st.session_state.initialized:
        initialize_chat_service()
    
    render_sidebar()
    render_chat_interface()


if __name__ == "__main__":
    main()

