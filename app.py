import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
import json
from datetime import datetime

# Import from semantic_search
from semantic_search import (
    load_documents,
    create_index,
    save_index,
    load_index,
    ConversationMemory,
    rewrite_query,
    hybrid_search,
    get_query_embedding,
    cosine_similarity,
    generate_answer_streaming
)

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Page config
st.set_page_config(
    page_title="RAG Chat System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    
    .stTextInput > div > div > input {
        background-color: #ffffff;
        border: 2px solid #e0e0e0;
        border-radius: 20px;
        padding: 10px 20px;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }
    
    .stButton > button {
        border-radius: 10px;
        border: none;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 8px 20px;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #667eea;
        font-weight: bold;
    }
    
    .streamlit-expanderHeader {
        background-color: #fff3e0;
        border-radius: 5px;
        font-weight: 600;
    }
    
    .stChatMessage {
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'docs' not in st.session_state:
    st.session_state.docs = None
if 'conversation' not in st.session_state:
    st.session_state.conversation = ConversationMemory(max_history=5)
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.title("🤖 RAG Chat System")
    st.markdown("Powered by HuggingFace Embeddings")
    st.markdown("---")
    
    # Document Management
    st.subheader("📚 Documents")
    
    docs_folder = st.text_input("Folder Path", value="docs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Load"):
            if Path(docs_folder).exists():
                with st.spinner("Loading..."):
                    docs = load_documents(Path(docs_folder))
                    docs = create_index(docs)
                    st.session_state.docs = docs
                    save_index(docs, "index_hf.json")  # Changed to HF index
                st.success(f"✅ {len(docs)} chunks")
            else:
                st.error(f"❌ Not found")
    
    with col2:
        if st.button("📂 Load Index"):
            if Path("index_hf.json").exists():  # Prefer HF index
                with st.spinner("Loading..."):
                    st.session_state.docs = load_index("index_hf.json")
                st.success(f"✅ {len(st.session_state.docs)} chunks")
            elif Path("index_streamlit.json").exists():  # For backward compatibility
                with st.spinner("Loading..."):
                    st.session_state.docs = load_index("index_streamlit.json")
                st.success(f"✅ {len(st.session_state.docs)} chunks")
            else:
                st.error("❌ No index")
    
    # File Upload
    st.markdown("---")
    st.subheader("📤 Upload Files")
    
    uploaded_files = st.file_uploader(
        "Drop files here",
        type=['txt', 'md'],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files and st.button("Process Uploads"):
        with st.spinner("Processing..."):
            temp_folder = Path("docs")
            temp_folder.mkdir(exist_ok=True)
            
            for uploaded_file in uploaded_files:
                file_path = temp_folder / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            docs = load_documents(temp_folder)
            
            if st.session_state.docs:
                existing_files = {d['filename'] for d in st.session_state.docs}
                new_docs = [d for d in docs if d['filename'] not in existing_files]
                
                if new_docs:
                    new_docs = create_index(new_docs)
                    st.session_state.docs.extend(new_docs)
                    st.success(f"✅ Added {len(new_docs)} chunks")
                else:
                    st.warning("Already indexed")
            else:
                docs = create_index(docs)
                st.session_state.docs = docs
                st.success(f"✅ {len(docs)} chunks")
            
            if st.session_state.docs:
                save_index(st.session_state.docs, "index_hf.json")  # Changed to HF index
    
    # Stats
    if st.session_state.docs:
        st.markdown("---")
        st.subheader("📊 Stats")
        
        col1, col2 = st.columns(2)
        with col1:
            total_docs = len(set(d['filename'] for d in st.session_state.docs))
            st.metric("Files", total_docs)
        with col2:
            st.metric("Chunks", len(st.session_state.docs))
        
        st.metric("Turns", len(st.session_state.conversation.history))
    
    # Settings
    st.markdown("---")
    st.subheader("⚙️ Settings")
    top_k = st.slider("Chunks", 1, 10, 3)
    use_hybrid = st.checkbox("Hybrid Search", value=True)
    
    # Actions
    st.markdown("---")
    st.subheader("🎬 Actions")
    
    if st.button("🆕 New Chat"):
        st.session_state.conversation.clear()
        st.session_state.messages = []
        st.rerun()
    
    if st.button("💾 Save"):
        if st.session_state.conversation.history:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"
            st.session_state.conversation.export(filename)
            st.success(f"Saved!")
        else:
            st.warning("Nothing to save")
    
    if st.button("📥 Download"):
        if st.session_state.messages:
            chat_export = {
                "exported_at": datetime.now().isoformat(),
                "total_messages": len(st.session_state.messages),
                "messages": st.session_state.messages
            }
            
            json_str = json.dumps(chat_export, indent=2, ensure_ascii=False)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            st.download_button(
                label="💾 Download JSON",
                data=json_str,
                file_name=f"chat_{timestamp}.json",
                mime="application/json",
                use_container_width=True
            )
        else:
            st.warning("No history")

# Main area
st.title("💬 RAG Chat Interface (HuggingFace Embeddings)")

# Tabs
tab1, tab2 = st.tabs(["💬 Chat", "📜 History"])

with tab1:
    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        chunk_info = f" [chunk {source['chunk_id']}]" if source['chunk_id'] > 0 else ""
                        st.markdown(f"**{i}.** `{source['filename']}{chunk_info}` - {source['score']:.3f}")
    
    # Chat input
    if prompt := st.chat_input("Ask about your documents..."):
        if not st.session_state.docs:
            st.error("⚠️ Load documents first")
            st.stop()
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            sources_container = st.container()
            
            conv_history = st.session_state.conversation.get_context()
            
            # Rewrite query
            rewritten_query = prompt
            if conv_history:
                with st.spinner("Understanding..."):
                    rewritten_query = rewrite_query(prompt, conv_history)
            
            # Retrieve
            with st.spinner("Searching..."):
                if use_hybrid:
                    results = hybrid_search(rewritten_query, st.session_state.docs, top_k=top_k)
                else:
                    query_emb = get_query_embedding(rewritten_query)
                    for doc in st.session_state.docs:
                        doc['score'] = cosine_similarity(query_emb, doc['embedding'])
                    results = sorted(st.session_state.docs, key=lambda x: x['score'], reverse=True)[:top_k]
            
            context_chunks = [r['text'] for r in results]
            sources = [{"filename": r['filename'], "chunk_id": r['chunk_id'], "score": r['score']} 
                       for r in results]
            
            # Stream answer
            full_answer = ""
            for chunk in generate_answer_streaming(prompt, context_chunks, conv_history, typing_speed=0.0):
                if isinstance(chunk, dict) and chunk.get("done"):
                    full_answer = chunk['full_answer']
                    st.session_state.conversation.add_turn(prompt, full_answer, sources)
                else:
                    full_answer += chunk
                    message_placeholder.markdown(full_answer + "▌")
            
            message_placeholder.markdown(full_answer)
            
            # Sources
            with sources_container.expander("📚 Sources"):
                for i, source in enumerate(sources, 1):
                    chunk_info = f" [chunk {source['chunk_id']}]" if source['chunk_id'] > 0 else ""
                    st.markdown(f"**{i}.** `{source['filename']}{chunk_info}` - {source['score']:.3f}")
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "sources": sources
            })

with tab2:
    st.subheader("📜 Conversation History")
    
    if st.session_state.conversation.history:
        for i, turn in enumerate(st.session_state.conversation.history, 1):
            with st.expander(f"Turn {i}: {turn['query'][:50]}...", expanded=(i == len(st.session_state.conversation.history))):
                st.markdown(f"**🕒** {turn['timestamp'][:19]}")
                st.markdown(f"**❓ Question:** {turn['query']}")
                st.markdown(f"**💡 Answer:** {turn['answer']}")
                
                st.markdown("**📚 Sources:**")
                for j, source in enumerate(turn['sources'], 1):
                    chunk_info = f" [chunk {source['chunk_id']}]" if source['chunk_id'] > 0 else ""
                    st.markdown(f"{j}. `{source['filename']}{chunk_info}` - {source['score']:.3f}")
    else:
        st.info("No history yet. Start chatting!")