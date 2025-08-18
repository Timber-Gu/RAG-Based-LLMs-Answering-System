"""
Streamlit Web Interface for LangChain ML Q&A Assistant
Modern chat interface with multi-agent system integration
"""
import streamlit as st
import os
import sys
from datetime import datetime
import json
from dotenv import load_dotenv

# Load environment variables with cache clearing
import os
# Clear any cached environment variables to force reload
if hasattr(os, 'unsetenv'):
    for key in list(os.environ.keys()):
        if key.startswith(('THEORY_MODEL', 'RESEARCH_MODEL', 'IMPLEMENTATION_MODEL')):
            try:
                del os.environ[key]
            except KeyError:
                pass

load_dotenv(override=True)

# Add src to path
sys.path.append('src')

# Page configuration
st.set_page_config(
    page_title="🤖 ML Q&A Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .agent-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.2rem;
    }
    .research-badge { background-color: #e3f2fd; color: #1976d2; }
    .theory-badge { background-color: #f3e5f5; color: #7b1fa2; }
    .implementation-badge { background-color: #e8f5e8; color: #388e3c; }
    .thinking-box {
        background-color: #f8f9fa;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .ai-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
    }
    .status-good { color: #4caf50; }
    .status-warning { color: #ff9800; }
    .status-error { color: #f44336; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_agents():
    """Initialize the LangChain agents with enhanced memory system (cached for performance)"""
    try:
        from src.agents.langchain_agents import create_langchain_ml_agents
        import uuid
        
        # Check for required API keys
        missing_keys = []
        if not os.getenv('OPENAI_API_KEY'):
            missing_keys.append('OPENAI_API_KEY')
        
        if missing_keys:
            st.error(f"❌ Missing required environment variables: {', '.join(missing_keys)}")
            st.stop()
        
        # Generate session ID for persistent memory
        if "session_id" not in st.session_state:
            st.session_state.session_id = f"streamlit_{uuid.uuid4().hex[:8]}"
        
        with st.spinner("🚀 Initializing LangChain agents with enhanced memory..."):
            agents = create_langchain_ml_agents(
                memory_type="auto",  # Automatic memory optimization
                session_id=st.session_state.session_id,
                enable_persistent_memory=True
            )
        
        # Warn if Pinecone is not configured (RAG features will be disabled)
        if not os.getenv('PINECONE_API_KEY'):
            st.warning("⚠️ PINECONE_API_KEY missing. Vector store features (upload/search) will be disabled.")
        
        return agents
    except Exception as e:
        st.error(f"❌ Error initializing agents: {e}")
        st.stop()

def sync_memory_with_session_state(agents):
    """Synchronize LangChain memory with Streamlit session state"""
    try:
        if not hasattr(agents, 'memory_manager') or not agents.memory_manager:
            # Fallback to legacy memory if enhanced memory not available
            if "chat_messages" not in st.session_state:
                st.session_state.chat_messages = []
            
            # Sync with legacy chat history
            legacy_history = agents.get_chat_history()
            if legacy_history:
                st.session_state.chat_messages = []
                for entry in legacy_history:
                    msg_type = entry['type']
                    content = entry['content']
                    
                    # Extract agent name from AI messages
                    agent_name = None
                    if msg_type == 'ai' and content.startswith('[') and '] ' in content:
                        try:
                            agent_name = content.split('] ')[0][1:].replace(' Agent', '').lower()
                            content = content.split('] ', 1)[1]
                        except:
                            pass
                    
                    st.session_state.chat_messages.append({
                        "type": msg_type,
                        "content": content,
                        "agent_name": agent_name
                    })
            return
        
        # Enhanced memory system synchronization
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        
        # Get memory statistics to check if there's conversation history
        stats = agents.get_memory_statistics()
        
        if stats.get("error") or stats.get("total_messages", 0) == 0:
            # No memory history, keep current session state
            return
        
        # Sync session state with enhanced memory
        # Since enhanced memory is handled automatically by agents,
        # we only need to ensure session state reflects the conversation
        
        # If session state is empty but memory has messages, rebuild session state
        if len(st.session_state.chat_messages) == 0 and stats.get("total_messages", 0) > 0:
            # Try to get conversation from any agent memory
            for agent_name, memory in agents.agent_memories.items():
                if agent_name == "semantic":
                    continue
                    
                try:
                    if hasattr(memory, 'chat_memory') and hasattr(memory.chat_memory, 'messages'):
                        messages = memory.chat_memory.messages[-10:]  # Get last 10 messages
                        
                        st.session_state.chat_messages = []
                        for msg in messages:
                            if hasattr(msg, 'content'):
                                if msg.__class__.__name__ == 'HumanMessage':
                                    st.session_state.chat_messages.append({
                                        "type": "user",
                                        "content": msg.content
                                    })
                                elif msg.__class__.__name__ == 'AIMessage':
                                    content = msg.content
                                    agent_name = None
                                    
                                    # Extract agent name from message
                                    if content.startswith('[') and '] ' in content:
                                        try:
                                            agent_name = content.split('] ')[0][1:].replace(' Agent', '').lower()
                                            content = content.split('] ', 1)[1]
                                        except:
                                            pass
                                    
                                    st.session_state.chat_messages.append({
                                        "type": "ai", 
                                        "content": content,
                                        "agent_name": agent_name
                                    })
                        break
                except Exception as e:
                    print(f"Warning: Could not sync memory from {agent_name}: {e}")
                    continue
                    
    except Exception as e:
        print(f"Warning: Memory synchronization failed: {e}")
        # Ensure session state is initialized even if sync fails
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

def display_system_status(agents):
    """Display system health and status"""
    st.subheader("🔍 System Status")
    
    health = agents.health_check()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🤖 Agents**")
        for agent in agents.get_available_agents():
            st.markdown(f"✅ {agent.capitalize()}")
    
    with col2:
        st.markdown("**🧠 Models**")
        if 'agent_models' in health:
            for agent, model in health['agent_models'].items():
                st.markdown(f"• **{agent.capitalize()}**: {model}")
    
    with col3:
        st.markdown("**🔗 Connections**")
        connections = {
            'GPT-4': health.get('gpt4_connection', False),
            'Claude': health.get('claude_connection', False),
            'Routing LLM': health.get('routing_llm_connection', False)
        }
        
        for service, status in connections.items():
            icon = "✅" if status else "❌"
            st.markdown(f"{icon} {service}")
    
    # Vector store status
    st.markdown("**📚 Vector Store**")
    vector_status = "✅ Connected" if health.get('vector_store') else "❌ Not available"
    st.markdown(f"• {vector_status} ({health.get('vector_store_type', 'Unknown')})")

def display_agent_badge(agent_name):
    """Display styled agent badge"""
    badge_class = f"{agent_name.lower()}-badge"
    return f'<span class="agent-badge {badge_class}">{agent_name.upper()}</span>'

def display_thinking_process(thinking_steps):
    """Display the agent's thinking process"""
    if not thinking_steps:
        return
    
    st.markdown('<div class="thinking-box">', unsafe_allow_html=True)
    st.markdown("### 🧠 Agent Thinking Process")
    
    for step in thinking_steps:
        with st.expander(f"Step {step['step_number']}: {step['description']}", expanded=False):
            if step.get('tool_name'):
                st.markdown(f"**Tool Used:** `{step['tool_name']}`")
            if step.get('tool_input'):
                st.markdown(f"**Input:** {step['tool_input']}")
            if step.get('result_summary'):
                st.markdown(f"**Result:** {step['result_summary']}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_chat_message(message_type, content, agent_name=None):
    """Display a formatted chat message"""
    if message_type == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🤔 You:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        agent_badge = display_agent_badge(agent_name) if agent_name else ""
        st.markdown(f"""
        <div class="chat-message ai-message">
            <strong>🤖 AI {agent_badge}:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)

def extract_text_from_pdf_bytes(pdf_bytes):
    """Extract text from PDF bytes using PyMuPDF"""
    try:
        import fitz  # PyMuPDF
        import re
        
        text = ""
        with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
            for page in pdf:
                text += page.get_text()
        
        # Clean up text
        text = re.sub(r'\n+', '\n', text)  # Remove multiple newlines
        text = re.sub(r'\s+', ' ', text)   # Normalize whitespace
        return text.strip()
        
    except ImportError:
        st.error("❌ PyMuPDF not installed. Please install with: pip install PyMuPDF")
        return ""
    except Exception as e:
        st.error(f"❌ Error extracting text from PDF: {e}")
        return ""

def create_paragraph_chunks(text, title, paper_id, source, authors, categories, max_chunk_size=1200, overlap_size=150):
    """Create chunks based on paragraph boundaries with intelligent merging"""
    import re
    
    # Use sentence-aware chunking for better content integrity
    sentence_endings = r'[.!?]+(?:\s|$)'
    sentences = re.split(sentence_endings, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Add sentence ending back (except for last sentence)
        if sentence != sentences[-1] and not sentence.endswith(('.', '!', '?')):
            sentence += '.'
        
        # Check if adding this sentence would exceed max size
        test_chunk = current_chunk + (' ' if current_chunk else '') + sentence
        
        if len(test_chunk) > max_chunk_size and current_chunk:
            # Current chunk is complete, save it
            chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap if specified
            if overlap_size > 0 and len(current_chunk) > overlap_size:
                # Find good overlap point (complete sentences only)
                words = current_chunk.split()
                overlap_text = ""
                for i in range(len(words) - 1, 0, -1):
                    candidate = ' '.join(words[i:])
                    if len(candidate) <= overlap_size and candidate.endswith(('.', '!', '?')):
                        overlap_text = candidate
                        break
                current_chunk = overlap_text + (' ' if overlap_text else '') + sentence
            else:
                current_chunk = sentence
        else:
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += ' ' + sentence
            else:
                current_chunk = sentence
    
    # Add the last chunk if it exists
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Create chunk entries
    chunk_entries = []
    for i, chunk in enumerate(chunks):
        if chunk.strip():  # Only add non-empty chunks
            chunk_entry = {
                "id": f"{paper_id}_chunk_{i+1}",
                "title": f"{title} (Part {i+1}/{len(chunks)})",
                "content": chunk.strip(),
                "source": source,
                "authors": authors,
                "categories": categories,
                "type": "chunk",
                "chunk_index": i + 1,
                "total_chunks": len(chunks),
                "parent_paper_id": paper_id
            }
            chunk_entries.append(chunk_entry)
    
    return chunk_entries

def extract_paper_metadata(text, filename):
    """Extract basic metadata from paper text"""
    lines = text.split('\n')
    
    # Try to find title (usually one of the first few lines)
    title = "Unknown Title"
    for line in lines[:10]:
        line = line.strip()
        if len(line) > 20 and not line.isupper() and not line.startswith(('arXiv:', 'http')):
            title = line
            break
    
    # Extract paper ID from filename
    paper_id = filename.replace('.pdf', '')
    
    # Try to extract authors (look for common patterns)
    authors = ["Unknown Author"]  # Default fallback
    
    # Infer categories based on content or filename patterns
    categories = ["cs.AI", "cs.ML"]  # Default ML/AI categories
    
    return {
        "title": title,
        "authors": authors,
        "categories": categories,
        "arxiv_id": paper_id,
        "source": f"Uploaded: {filename}"
    }

def process_uploaded_paper(uploaded_file, agents, chunking_method="paragraph"):
    """Process uploaded PDF paper and upsert to Pinecone"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Extract text from PDF
        status_text.text("📖 Extracting text from PDF...")
        progress_bar.progress(20)
        
        pdf_bytes = uploaded_file.read()
        text = extract_text_from_pdf_bytes(pdf_bytes)
        
        if not text:
            st.error("❌ Failed to extract text from PDF")
            return False
        
        st.info(f"✅ Extracted {len(text)} characters of text")
        
        # Step 2: Extract metadata
        status_text.text("🏷️ Extracting metadata...")
        progress_bar.progress(40)
        
        filename = uploaded_file.name
        metadata = extract_paper_metadata(text, filename)
        
        st.info(f"📝 Title: {metadata['title'][:100]}...")
        
        # Step 3: Create chunks
        status_text.text("✂️ Creating intelligent chunks...")
        progress_bar.progress(60)
        
        if chunking_method == "paragraph":
            chunks = create_paragraph_chunks(
                text=text,
                title=metadata['title'],
                paper_id=metadata['arxiv_id'],
                source=metadata['source'],
                authors=metadata['authors'],
                categories=metadata['categories'],
                max_chunk_size=1200,
                overlap_size=150
            )
        else:  # traditional
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from src.config import settings
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            # Split text and create chunks
            text_chunks = splitter.split_text(text)
            chunks = []
            for i, chunk in enumerate(text_chunks):
                chunk_entry = {
                    "id": f"{metadata['arxiv_id']}_chunk_{i+1}",
                    "title": f"{metadata['title']} (Part {i+1}/{len(text_chunks)})",
                    "content": chunk,
                    "source": metadata['source'],
                    "authors": metadata['authors'],
                    "categories": metadata['categories'],
                    "type": "chunk",
                    "chunk_index": i + 1,
                    "total_chunks": len(text_chunks),
                    "parent_paper_id": metadata['arxiv_id']
                }
                chunks.append(chunk_entry)
        
        st.info(f"📚 Created {len(chunks)} chunks")
        
        # Step 4: Convert to LangChain Documents
        status_text.text("🔄 Converting to documents...")
        progress_bar.progress(70)
        
        from langchain_core.documents import Document
        
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk['content'],
                metadata={
                    'title': chunk['title'],
                    'source': chunk['source'],
                    'category': 'research_paper',
                    'type': chunk['type'],
                    'id': chunk['id'],
                    'authors': chunk['authors'],
                    'categories': chunk['categories'],
                    'chunk_index': chunk['chunk_index'],
                    'total_chunks': chunk['total_chunks'],
                    'parent_paper_id': chunk['parent_paper_id']
                }
            )
            documents.append(doc)
        
        # Step 5: Upsert to Pinecone
        status_text.text("⬆️ Uploading to Pinecone...")
        progress_bar.progress(90)
        
        if not agents.pinecone_index:
            st.error("❌ Pinecone not connected. Check your API configuration.")
            return False
        
        success = agents.upsert_documents_to_pinecone(documents)
        
        if success:
            # Step 6: Update knowledge base JSON
            status_text.text("💾 Updating knowledge base...")
            progress_bar.progress(95)
            
            from src.config import settings
            knowledge_base_path = settings.KNOWLEDGE_BASE_FILE
            knowledge_base = []
            
            if os.path.exists(knowledge_base_path):
                with open(knowledge_base_path, 'r', encoding='utf-8') as f:
                    knowledge_base = json.load(f)
            
            # Remove existing chunks for this paper to avoid duplicates
            knowledge_base = [entry for entry in knowledge_base 
                            if entry.get('parent_paper_id') != metadata['arxiv_id']]
            
            # Add new chunks
            knowledge_base.extend(chunks)
            
            # Save updated knowledge base
            with open(knowledge_base_path, 'w', encoding='utf-8') as f:
                json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
            
            progress_bar.progress(100)
            status_text.text("🎉 Successfully processed and uploaded!")
            
            st.success(f"✅ Successfully processed **{filename}**")
            st.info(f"📊 **{len(chunks)} chunks** added to your knowledge base")
            st.info(f"🔍 You can now ask questions about this paper!")
            
            return True
        else:
            st.error("❌ Failed to upload to Pinecone")
            return False
            
    except Exception as e:
        st.error(f"❌ Error processing paper: {e}")
        return False
    finally:
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 LangChain ML Q&A Assistant</h1>
        <p>Multi-Agent System for Machine Learning & Deep Learning Questions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize agents
    agents = initialize_agents()
    
    # Synchronize memory with session state
    sync_memory_with_session_state(agents)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Agent selection mode
        routing_mode = st.radio(
            "🎯 Routing Mode",
            ["Automatic (Smart Routing)", "Manual Selection"],
            help="Choose how queries are routed to agents"
        )
        
        if routing_mode == "Manual Selection":
            selected_agent = st.selectbox(
                "Choose Agent",
                agents.get_available_agents(),
                format_func=lambda x: f"{x.capitalize()} Agent"
            )
        
        # Display settings
        show_thinking = st.checkbox("🧠 Show Thinking Process", value=True)
        show_system_status = st.checkbox("🔍 Show System Status", value=False)
        
        st.divider()
        
        # Enhanced Memory Management
        st.header("🧠 Enhanced Memory")
        
        # Show memory statistics
        if hasattr(agents, 'memory_manager') and agents.memory_manager:
            stats = agents.get_memory_statistics()
            if not stats.get("error"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Messages", stats.get('total_messages', 0))
                with col2:
                    st.metric("Efficiency", f"{stats.get('memory_efficiency_score', 0):.0f}%")
                
                # Memory type and session info
                st.caption(f"🆔 Session: {agents.session_id}")
                st.caption(f"🧠 Type: {agents.memory_type}")
                st.caption(f"⏱️ Duration: {stats.get('conversation_duration_minutes', 0):.1f}min")
                
                # Optimization recommendations
                if stats.get('optimization_recommendations'):
                    with st.expander("💡 Recommendations"):
                        for rec in stats['optimization_recommendations']:
                            st.write(f"• {rec}")
            else:
                st.info("Memory statistics not available")
        else:
            # Fallback to legacy memory
            history_summary = agents.get_chat_history_summary()
            st.metric("Messages", history_summary['total_messages'])
            st.caption("Using legacy memory system")
        
        # Memory management buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear", use_container_width=True):
                if hasattr(agents, 'memory_manager') and agents.memory_manager:
                    result = agents.clear_enhanced_memory(confirm=True)
                    if result.get("clear_successful"):
                        st.session_state.chat_messages = []  # Clear session state too
                        st.success("✅ Cleared!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to clear")
                else:
                    agents.clear_chat_history()
                    st.session_state.chat_messages = []
                    st.success("✅ Cleared!")
                    st.rerun()
        
        with col2:
            if st.button("🔧 Optimize", use_container_width=True):
                if hasattr(agents, 'memory_manager') and agents.memory_manager:
                    result = agents.optimize_memory_settings()
                    if result.get("optimization_successful"):
                        st.success("✅ Optimized!")
                        if result.get('optimizations_applied'):
                            st.info(f"Applied: {', '.join(result['optimizations_applied'])}")
                    else:
                        st.info("No optimization needed")
                else:
                    st.info("Enhanced memory not available")
        
        with col3:
            if st.button("📸 Snapshot", use_container_width=True):
                if hasattr(agents, 'memory_manager') and agents.memory_manager:
                    result = agents.create_memory_snapshot()
                    if result.get("snapshot_successful"):
                        st.success("✅ Snapshot created!")
                        st.caption(f"Name: {result['snapshot_name']}")
                    else:
                        st.error("❌ Snapshot failed")
                else:
                    st.info("Enhanced memory not available")
        
        # Memory type switching
        if hasattr(agents, 'memory_manager') and agents.memory_manager:
            with st.expander("🔄 Memory Settings"):
                current_type = getattr(agents, 'memory_type', 'unknown')
                new_memory_type = st.selectbox(
                    "Memory Type",
                    ["summary_buffer", "buffer_window", "token_buffer", "auto"],
                    index=["summary_buffer", "buffer_window", "token_buffer", "auto"].index(current_type) if current_type in ["summary_buffer", "buffer_window", "token_buffer", "auto"] else 0,
                    help="Change memory strategy"
                )
                
                if st.button("🔄 Switch Memory Type") and new_memory_type != current_type:
                    result = agents.switch_memory_type(new_memory_type, preserve_history=True)
                    if result.get("switch_successful"):
                        st.success(f"✅ Switched to {new_memory_type}")
                        st.rerun()
                    else:
                        st.error("❌ Switch failed")
        
        # Legacy save/load for compatibility
        with st.expander("💾 Legacy Save/Load"):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save JSON", use_container_width=True):
                    if agents.save_chat_history_to_file("streamlit_chat_history.json"):
                        st.success("✅ Saved!")
                    else:
                        st.error("❌ Failed to save")
            
            with col2:
                if st.button("📁 Load JSON", use_container_width=True):
                    if agents.load_chat_history_from_file("streamlit_chat_history.json"):
                        st.success("✅ Loaded!")
                        sync_memory_with_session_state(agents)  # Re-sync after load
                        st.rerun()
                    else:
                        st.error("❌ Failed to load")
        
        st.divider()
        
        # Vector store management
        st.header("📚 Knowledge Base")
        
        # Paper Upload Section
        st.subheader("📄 Upload New Paper")
        uploaded_file = st.file_uploader(
            "Choose a PDF paper",
            type=['pdf'],
            help="Upload a research paper to process and add to your knowledge base"
        )
        
        if uploaded_file is not None:
            # File details
            st.write(f"**File:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
            
            # Processing options
            chunking_method = st.selectbox(
                "Chunking Method",
                ["paragraph", "traditional"],
                format_func=lambda x: "Paragraph-based (Recommended)" if x == "paragraph" else "Traditional Recursive",
                help="Choose how to split the paper into chunks"
            )
            
            # Process button
            if st.button("🔄 Process & Upload Paper", use_container_width=True):
                with st.spinner("Processing paper and uploading to Pinecone..."):
                    success = process_uploaded_paper(uploaded_file, agents, chunking_method)
                    if success:
                        st.success("✅ Paper processed and uploaded successfully!")
                        st.balloons()
                        # Clear the file uploader
                        st.rerun()
                    else:
                        st.error("❌ Failed to process paper")
        
        st.divider()
        
        # Existing upload functionality
        if st.button("⬆️ Upload Existing Knowledge Base", use_container_width=True, help="Upload existing knowledge base to Pinecone vector store"):
            if not getattr(agents, 'pinecone_index', None):
                st.warning("⚠️ Pinecone is not connected. Set PINECONE_API_KEY and restart.")
            else:
                with st.spinner("Uploading to Pinecone..."):
                    success = agents.upsert_knowledge_base_to_pinecone()
                    if success:
                        st.success("✅ Upload successful!")
                    else:
                        st.error("❌ Upload failed")
    
    # Main content area
    main_col, status_col = st.columns([3, 1]) if show_system_status else (st.container(), None)
    
    with main_col:
        # Chat interface
        st.header("💬 Chat Interface")
        
        # Initialize session state for chat history (synchronized with LangChain memory)
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        
        # Display chat history from session state
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_messages:
                display_chat_message(
                    message["type"], 
                    message["content"], 
                    message.get("agent_name")
                )
        
        # Show memory status for debugging
        if hasattr(agents, 'memory_manager') and agents.memory_manager:
            stats = agents.get_memory_statistics()
            if not stats.get("error") and stats.get("total_messages", 0) > 0:
                st.caption(f"🧠 Memory: {stats.get('total_messages')} total messages | Efficiency: {stats.get('memory_efficiency_score', 0):.0f}%")
        
        # Query input
        query = st.chat_input("Ask your ML/DL question here...")
        
        if query:
            # Add user message to session state
            st.session_state.chat_messages.append({
                "type": "user",
                "content": query
            })
            
            # Display user message immediately
            display_chat_message("user", query)
            
            # Process query
            with st.spinner("🔄 Processing your question..."):
                try:
                    # Override agent routing if manual selection is enabled
                    if routing_mode == "Manual Selection":
                        # Temporarily override the routing method
                        original_route_method = agents.route_query
                        agents.route_query = lambda q: selected_agent
                    
                    result = agents.process_query(query, show_thinking=show_thinking)
                    
                    # Restore original routing method if it was overridden
                    if routing_mode == "Manual Selection":
                        agents.route_query = original_route_method
                    
                    if result.get('success'):
                        agent_used = result['agent_used']
                        response = result['response']
                        
                        # Display thinking process if available
                        if show_thinking and result.get('has_thinking') and result.get('thinking_process'):
                            display_thinking_process(result['thinking_process'])
                        
                        # Display AI response
                        display_chat_message("ai", response, agent_used)
                        
                        # Add AI message to session state
                        st.session_state.chat_messages.append({
                            "type": "ai",
                            "content": response,
                            "agent_name": agent_used
                        })
                        
                        # Show agent info
                        st.info(f"🤖 Response from: {display_agent_badge(agent_used)}", icon="ℹ️")
                        
                        # Re-synchronize memory with session state after processing
                        # This ensures the session state reflects any memory updates
                        try:
                            if hasattr(agents, 'memory_manager') and agents.memory_manager:
                                # Force a brief sync to ensure session state is up to date
                                pass  # The memory is already handled by the agents automatically
                        except Exception as sync_error:
                            print(f"Warning: Post-query sync failed: {sync_error}")
                        
                    else:
                        error_msg = result.get('error', 'Unknown error occurred')
                        st.error(f"❌ Error: {error_msg}")
                        
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
                
                # After processing any query, ensure memory is synced
                # This handles both success and error cases
                try:
                    if hasattr(agents, 'memory_manager') and agents.memory_manager:
                        stats = agents.get_memory_statistics()
                        if not stats.get("error"):
                            # Update session info display
                            st.caption(f"🔄 Memory updated: {stats.get('total_messages', 0)} messages")
                except Exception as e:
                    pass  # Ignore sync errors for display
    
    # System status column
    if show_system_status and status_col:
        with status_col:
            display_system_status(agents)
    
    # Help section
    with st.expander("❓ How to Use", expanded=False):
        st.markdown("""
        ### 🚀 Getting Started
        
        1. **Ask Questions**: Type your ML/DL questions in the chat input
        2. **Choose Routing**: Use automatic smart routing or manually select an agent
        3. **View Thinking**: Enable thinking process to see how agents reason
        
        ### 🤖 Available Agents
        
        - **🔬 Research Agent**: Literature reviews, recent papers, research trends
        - **📚 Theory Agent**: Mathematical concepts, theoretical explanations, algorithms  
        - **💻 Implementation Agent**: Code examples, practical tutorials, debugging
        
        ### 💡 Example Questions
        
        - *"What are the latest developments in transformer architectures?"* → Research Agent
        - *"Explain how backpropagation works mathematically"* → Theory Agent  
        - *"Show me how to implement a CNN in PyTorch"* → Implementation Agent
        
        ### ⚙️ Features
        
        - **Smart Routing**: Automatically routes questions to the best agent
        - **RAG Integration**: Searches knowledge base for relevant information
        - **Paper Upload**: Upload PDFs directly to expand knowledge base
        - **Intelligent Chunking**: Advanced paragraph-based text splitting
        - **Chat Memory**: Maintains conversation context
        - **Thinking Process**: Shows agent reasoning steps
        - **Multi-Model Support**: Uses GPT-4, Claude, and other models
        
        ### 📄 Paper Upload Guide
        
        1. **Upload PDF**: Use the file uploader in the sidebar
        2. **Choose Chunking**: Select paragraph-based (recommended) or traditional
        3. **Process**: Click "Process & Upload Paper" to add to knowledge base
        4. **Ask Questions**: The paper content is now searchable in your RAG system
        """)

if __name__ == "__main__":
    main() 