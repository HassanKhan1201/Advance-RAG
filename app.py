import streamlit as st
import os
import time
import json
from datetime import datetime
import plotly.express as px
import pandas as pd

# Import our custom modules
from utils.processing_ocr import DocumentProcessor
from utils.embedding import AdvancedEmbeddingManager
from utils.llm import LLMResponseManager

# Page configuration
st.set_page_config(
    page_title="Advanced RAG Document Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #dee2e6;
        margin: 1rem 0;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .user-message {
        background: #e3f2fd;
        border-left-color: #2196f3;
    }
    
    .assistant-message {
        background: #f3e5f5;
        border-left-color: #9c27b0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .status-indicator {
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .status-success {
        background: #d4edda;
        color: #155724;
    }
    
    .status-warning {
        background: #fff3cd;
        color: #856404;
    }
    
    .status-error {
        background: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "processing_status" not in st.session_state:
    st.session_state.processing_status = "idle"
if "embedding_stats" not in st.session_state:
    st.session_state.embedding_stats = {}

# Configuration
UPLOAD_FOLDER = "uploads"
TEXT_FOLDER = "extracted_text"
FAISS_FOLDER = "faiss_index"
METADATA_FILE = os.path.join(FAISS_FOLDER, "metadata.json")

# Ensure directories exist
for folder in [UPLOAD_FOLDER, TEXT_FOLDER, FAISS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize RAG components with caching."""
    doc_processor = DocumentProcessor()
    embedding_manager = AdvancedEmbeddingManager()
    llm_manager = LLMResponseManager()
    return doc_processor, embedding_manager, llm_manager

doc_processor, embedding_manager, llm_manager = initialize_components()

def main():
    """Main application function."""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Advanced RAG Document Analyzer</h1>
        <p>Upload documents and ask intelligent questions using AI-powered retrieval</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # Model selection
        st.subheader("Model Settings")
        available_models = llm_manager.get_available_models()
        selected_model = st.selectbox(
            "Select LLM Model",
            options=list(available_models.keys()),
            format_func=lambda x: f"{x}\n{available_models[x]}",
            help="Choose the language model for generating responses"
        )
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            prompt_type = st.selectbox(
                "Prompt Strategy",
                ["base", "few_shot", "zero_shot", "role_specific", "conversational"],
                help="Select the prompting technique"
            )
            
            use_mmr = st.toggle(
                "Use MMR (Maximal Marginal Relevance)",
                value=True,
                help="Enable MMR for diverse search results"
            )
            
            top_k = st.slider(
                "Number of Retrieved Chunks",
                min_value=3,
                max_value=10,
                value=5,
                help="Number of relevant text chunks to retrieve"
            )
        
        # Document statistics
        if st.session_state.embedding_stats:
            st.subheader("📊 Document Statistics")
            stats = st.session_state.embedding_stats
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Chunks", stats.get("total_chunks", 0))
                st.metric("Pages", stats.get("total_pages", 0))
            with col2:
                st.metric("Vectors", stats.get("total_vectors", 0))
                st.metric("Dimension", stats.get("vector_dimension", 0))
            
            if stats.get("file_types"):
                st.write("**File Types:**", ", ".join(stats["file_types"]))
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📄 Document Upload", "💬 Chat Interface", "📈 Analytics"])
    
    with tab1:
        document_upload_interface()
    
    with tab2:
        if st.session_state.document_processed:
            chat_interface(selected_model, prompt_type, use_mmr, top_k)
        else:
            st.info("👆 Please upload and process a document first to enable the chat interface.")
    
    with tab3:
        analytics_interface()

def document_upload_interface():
    """Document upload and processing interface."""
    
    st.header("📄 Document Upload & Processing")
    
    # File upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a document to analyze",
        type=['pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif', 'docx', 'doc'],
        help="Supported formats: PDF, Images (PNG, JPG, etc.), Word documents"
    )
    
    if uploaded_file is not None:
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Filename:** {uploaded_file.name}")
        with col2:
            st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
        with col3:
            file_type = doc_processor.get_file_type(uploaded_file.name)
            st.write(f"**Type:** {file_type.upper() if file_type else 'Unknown'}")
        
        # Process button
        if st.button("🚀 Process Document", type="primary", use_container_width=True):
            process_document(uploaded_file)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Processing status
    if st.session_state.processing_status != "idle":
        display_processing_status()

def process_document(uploaded_file):
    """Process uploaded document through the RAG pipeline."""
    
    try:
        st.session_state.processing_status = "processing"
        
        # Step 1: Document Processing (OCR)
        st.info("🔄 Step 1: Processing document and extracting text...")
        metadata = doc_processor.process_document(uploaded_file)
        
        if not metadata:
            st.error("❌ Failed to process document. Please try again.")
            st.session_state.processing_status = "error"
            return
        
        # Save metadata
        metadata_file = os.path.join(TEXT_FOLDER, "metadata.json")
        if not doc_processor.save_metadata(metadata, metadata_file):
            st.error("❌ Failed to save document metadata.")
            st.session_state.processing_status = "error"
            return
        
        # Step 2: Generate Embeddings
        st.info("🔄 Step 2: Generating embeddings and building vector database...")
        success = embedding_manager.generate_embeddings_and_store(metadata, FAISS_FOLDER)
        
        if not success:
            st.error("❌ Failed to generate embeddings.")
            st.session_state.processing_status = "error"
            return
        
        # Update session state
        st.session_state.document_processed = True
        st.session_state.processing_status = "completed"
        st.session_state.embedding_stats = embedding_manager.get_embedding_stats(FAISS_FOLDER)
        
        # Display success message
        st.success("✅ Document processed successfully! You can now ask questions about the document.")
        
        # Display processing summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Pages Processed", len(set(entry.get("page", 1) for entry in metadata)))
        with col2:
            st.metric("Text Chunks", st.session_state.embedding_stats.get("total_chunks", 0))
        with col3:
            st.metric("Embeddings Created", st.session_state.embedding_stats.get("total_vectors", 0))
        
        time.sleep(1)  # Brief pause to show completion
        
    except Exception as e:
        st.error(f"❌ Error processing document: {str(e)}")
        st.session_state.processing_status = "error"

def display_processing_status():
    """Display current processing status with indicators."""
    
    status = st.session_state.processing_status
    
    if status == "processing":
        st.markdown("""
        <div class="status-indicator status-warning">
            🔄 Processing in progress...
        </div>
        """, unsafe_allow_html=True)
        
    elif status == "completed":
        st.markdown("""
        <div class="status-indicator status-success">
            ✅ Processing completed successfully!
        </div>
        """, unsafe_allow_html=True)
        
    elif status == "error":
        st.markdown("""
        <div class="status-indicator status-error">
            ❌ Processing failed. Please try again.
        </div>
        """, unsafe_allow_html=True)

def chat_interface(selected_model, prompt_type, use_mmr, top_k):
    """Advanced chat interface for document querying."""
    
    st.header("💬 Intelligent Document Chat")
    
    # Quick query suggestions
    st.subheader("💡 Quick Query Suggestions")
    col1, col2, col3 = st.columns(3)
    
    suggestions = [
        "Summarize the main points of this document",
        "What are the key dates mentioned?",
        "Who are the main people or entities mentioned?",
        "What amounts or numbers are specified?",
        "What is the purpose of this document?",
        "Are there any important deadlines or dates?"
    ]
    
    for i, suggestion in enumerate(suggestions):
        col = [col1, col2, col3][i % 3]
        with col:
            if st.button(suggestion, key=f"suggestion_{i}", help="Click to use this query"):
                st.session_state.current_query = suggestion

    # Chat input
    query = st.chat_input(
        "Ask a question about your document...",
        key="chat_input"
    )
    
    # Use suggestion if selected
    if hasattr(st.session_state, 'current_query'):
        query = st.session_state.current_query
        delattr(st.session_state, 'current_query')
    
    # Process query
    if query:
        # Add user message to chat history
        st.session_state.chat_history.append({
            "type": "user",
            "content": query,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        # Analyze query complexity
        complexity_analysis = llm_manager.analyze_query_complexity(query)
        
        # Apply complexity-based suggestions
        suggested_params = complexity_analysis.get("suggestions", {})
        actual_prompt_type = suggested_params.get("prompt_type", prompt_type)
        actual_top_k = suggested_params.get("top_k", top_k)
        actual_use_mmr = suggested_params.get("use_mmr", use_mmr)
        
        # Show complexity indicator
        complexity_level = complexity_analysis.get("complexity_level", "medium")
        if complexity_level == "high":
            st.info(f"🧠 Detected high complexity query. Using advanced processing...")
        
        # Generate response
        with st.spinner("🤔 Thinking..."):
            response_data = llm_manager.query_model(
                query=query,
                faiss_folder=FAISS_FOLDER,
                metadata_file=METADATA_FILE,
                model_name=selected_model,
                prompt_type=actual_prompt_type,
                use_mmr=actual_use_mmr,
                top_k=actual_top_k,
                chat_history=st.session_state.chat_history
            )
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({
            "type": "assistant",
            "content": response_data["response"],
            "page": response_data.get("page"),
            "sources": response_data.get("sources", []),
            "confidence": response_data.get("confidence", "medium"),
            "processing_time": response_data.get("processing_time", 0),
            "model_used": response_data.get("model_used", selected_model),
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
    
    # Display chat history
    display_chat_history()
    
    # Chat controls
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🗑️ Clear Chat", help="Clear all chat history"):
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        if st.button("💾 Export Chat", help="Export chat history"):
            export_chat_history()
    
    with col3:
        if st.button("🔄 Reprocess Document", help="Reprocess the current document"):
            st.session_state.document_processed = False
            st.session_state.processing_status = "idle"
            st.rerun()

def display_chat_history():
    """Display chat messages with enhanced formatting."""
    
    if not st.session_state.chat_history:
        st.info("👋 Start a conversation by asking a question about your document!")
        return
    
    # Display messages
    for i, message in enumerate(st.session_state.chat_history):
        if message["type"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
                st.caption(f"⏰ {message['timestamp']}")
        
        else:  # assistant message
            with st.chat_message("assistant"):
                st.markdown(message["content"])
                
                # Show response metadata
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if message.get("page"):
                        st.caption(f"📄 Page {message['page']}")
                with col2:
                    confidence = message.get("confidence", "medium")
                    confidence_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "🟡")
                    st.caption(f"{confidence_emoji} {confidence.title()} confidence")
                with col3:
                    if message.get("processing_time"):
                        st.caption(f"⚡ {message['processing_time']}s")
                with col4:
                    st.caption(f"⏰ {message['timestamp']}")
                
                # Show sources if available
                if message.get("sources"):
                    with st.expander("📚 Sources", expanded=False):
                        for j, source in enumerate(message["sources"][:3]):  # Show top 3 sources
                            st.write(f"**Source {j+1}:** Page {source.get('page', 'Unknown')} "
                                   f"({source.get('file', 'Unknown file')})")
                            if source.get("similarity"):
                                st.progress(source["similarity"])

def analytics_interface():
    """Analytics and insights interface."""
    
    st.header("📈 Analytics & Insights")
    
    if not st.session_state.chat_history:
        st.info("📊 Analytics will be available after you start chatting with your document.")
        return
    
    # Chat statistics
    st.subheader("💬 Chat Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    user_messages = [msg for msg in st.session_state.chat_history if msg["type"] == "user"]
    assistant_messages = [msg for msg in st.session_state.chat_history if msg["type"] == "assistant"]
    
    with col1:
        st.metric("Total Questions", len(user_messages))
    with col2:
        st.metric("Total Responses", len(assistant_messages))
    with col3:
        avg_response_time = sum(msg.get("processing_time", 0) for msg in assistant_messages) / len(assistant_messages) if assistant_messages else 0
        st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
    with col4:
        high_confidence = sum(1 for msg in assistant_messages if msg.get("confidence") == "high")
        confidence_rate = (high_confidence / len(assistant_messages) * 100) if assistant_messages else 0
        st.metric("High Confidence Rate", f"{confidence_rate:.1f}%")
    
    # Response time chart
    if len(assistant_messages) > 1:
        st.subheader("⚡ Response Time Trends")
        
        df_times = pd.DataFrame([
            {
                "Message": i+1, 
                "Response Time (s)": msg.get("processing_time", 0),
                "Confidence": msg.get("confidence", "medium")
            }
            for i, msg in enumerate(assistant_messages)
        ])
        
        fig = px.line(df_times, x="Message", y="Response Time (s)", 
                     color="Confidence", title="Response Time by Message",
                     markers=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # Confidence distribution
    if assistant_messages:
        st.subheader("🎯 Confidence Distribution")
        
        confidence_counts = {}
        for msg in assistant_messages:
            conf = msg.get("confidence", "medium")
            confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
        
        df_confidence = pd.DataFrame([
            {"Confidence Level": k.title(), "Count": v} 
            for k, v in confidence_counts.items()
        ])
        
        fig = px.pie(df_confidence, values="Count", names="Confidence Level",
                    title="Response Confidence Distribution",
                    color_discrete_map={"High": "#00cc96", "Medium": "#ffa15a", "Low": "#ef553b"})
        st.plotly_chart(fig, use_container_width=True)
    
    # Page reference analysis
    page_refs = {}
    for msg in assistant_messages:
        page = msg.get("page")
        if page:
            page_refs[page] = page_refs.get(page, 0) + 1
    
    if page_refs:
        st.subheader("📄 Most Referenced Pages")
        
        df_pages = pd.DataFrame([
            {"Page": k, "References": v} 
            for k, v in sorted(page_refs.items())
        ])
        
        fig = px.bar(df_pages, x="Page", y="References",
                    title="Page Reference Frequency")
        st.plotly_chart(fig, use_container_width=True)
    
    # Export analytics
    if st.button("📊 Export Analytics Report"):
        export_analytics_report()

def export_chat_history():
    """Export chat history to JSON."""
    
    try:
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_messages": len(st.session_state.chat_history),
            "chat_history": st.session_state.chat_history,
            "document_stats": st.session_state.embedding_stats
        }
        
        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
        
        st.download_button(
            label="💾 Download Chat History",
            data=json_str,
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
    except Exception as e:
        st.error(f"Error exporting chat history: {e}")

def export_analytics_report():
    """Export analytics report."""
    
    try:
        user_messages = [msg for msg in st.session_state.chat_history if msg["type"] == "user"]
        assistant_messages = [msg for msg in st.session_state.chat_history if msg["type"] == "assistant"]
        
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_questions": len(user_messages),
                "total_responses": len(assistant_messages),
                "avg_response_time": sum(msg.get("processing_time", 0) for msg in assistant_messages) / len(assistant_messages) if assistant_messages else 0,
                "high_confidence_rate": sum(1 for msg in assistant_messages if msg.get("confidence") == "high") / len(assistant_messages) * 100 if assistant_messages else 0
            },
            "document_stats": st.session_state.embedding_stats,
            "detailed_analytics": {
                "confidence_distribution": {},
                "page_references": {},
                "response_times": [msg.get("processing_time", 0) for msg in assistant_messages]
            }
        }
        
        # Calculate distributions
        for msg in assistant_messages:
            conf = msg.get("confidence", "medium")
            report["detailed_analytics"]["confidence_distribution"][conf] = report["detailed_analytics"]["confidence_distribution"].get(conf, 0) + 1
            
            page = msg.get("page")
            if page:
                report["detailed_analytics"]["page_references"][str(page)] = report["detailed_analytics"]["page_references"].get(str(page), 0) + 1
        
        json_str = json.dumps(report, indent=2, ensure_ascii=False)
        
        st.download_button(
            label="📊 Download Analytics Report",
            data=json_str,
            file_name=f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
    except Exception as e:
        st.error(f"Error exporting analytics report: {e}")

if __name__ == "__main__":
    main()