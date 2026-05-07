"""
Streamlit Frontend for Real Estate RAG Assistant
"""
import streamlit as st
import os
from pathlib import Path
from document_loader import RealEstateDocumentLoader
from rag_backend import RealEstateRAG


# Page configuration
st.set_page_config(
    page_title="🏠 Real Estate RAG Assistant",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None

if "doc_loader" not in st.session_state:
    st.session_state.doc_loader = RealEstateDocumentLoader()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def initialize_rag():
    """Initialize the RAG system"""
    if st.session_state.rag_system is None:
        with st.spinner("🔄 Initializing RAG system..."):
            st.session_state.rag_system = RealEstateRAG(
                model_name="mistral",
                embedding_model="nomic-embed-text",
                vector_db_path="./chroma_db"
            )
            st.session_state.rag_system.load_vector_store()


def main():
    st.title("🏠 Real Estate RAG Assistant")
    st.markdown("*Powered by LangChain, Ollama, and Streamlit*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Initialize system
        if st.button("🚀 Initialize System", use_container_width=True):
            initialize_rag()
            st.success("System initialized!")
        
        st.divider()
        
        # Document management
        st.subheader("📄 Document Management")
        
        tab1, tab2, tab3 = st.tabs(["Upload PDF", "Load Listings", "Add Web Content"])
        
        with tab1:
            st.subheader("Upload PDF Documents")
            uploaded_files = st.file_uploader(
                "Upload property brochures, guides, or contracts (PDF)",
                type=["pdf"],
                accept_multiple_files=True
            )
            
            if uploaded_files and st.button("📤 Process PDFs", key="pdf_btn"):
                initialize_rag()
                all_docs = []
                
                for uploaded_file in uploaded_files:
                    # Save temporarily
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Load PDF
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        docs = st.session_state.doc_loader.load_pdf(temp_path)
                        all_docs.extend(docs)
                    
                    # Clean up
                    os.remove(temp_path)
                
                # Add to vector store
                if all_docs:
                    st.session_state.rag_system.add_documents(all_docs)
                    st.success(f"✅ Loaded {len(all_docs)} document chunks!")
        
        with tab2:
            st.subheader("Property Listings")
            listings_file = st.file_uploader(
                "Upload CSV with property listings",
                type=["csv"],
                key="csv_upload"
            )
            
            if listings_file and st.button("📤 Load Listings", key="csv_btn"):
                initialize_rag()
                temp_csv = "temp_listings.csv"
                with open(temp_csv, "wb") as f:
                    f.write(listings_file.getbuffer())
                
                with st.spinner("Processing listings..."):
                    docs = st.session_state.doc_loader.load_csv_listings(temp_csv)
                
                if docs:
                    st.session_state.rag_system.add_documents(docs)
                    st.success(f"✅ Loaded {len(docs)} listing chunks!")
                
                os.remove(temp_csv)
        
        with tab3:
            st.subheader("Web Content")
            url = st.text_input("Enter URL to scrape")
            
            if st.button("🌐 Load Web Content", key="web_btn"):
                if url:
                    initialize_rag()
                    with st.spinner(f"Scraping {url}..."):
                        docs = st.session_state.doc_loader.scrape_web_content(url)
                    
                    if docs:
                        st.session_state.rag_system.add_documents(docs)
                        st.success(f"✅ Loaded {len(docs)} content chunks!")
                else:
                    st.warning("Please enter a URL")
        
        st.divider()
        
        # Settings
        st.subheader("🔧 Settings")
        
        model_choice = st.selectbox(
            "Ollama Model",
            ["mistral", "neural-chat", "llama2", "orca-mini"]
        )
        
        if st.button("🗑️ Clear Vector Store", use_container_width=True):
            if st.session_state.rag_system:
                st.session_state.rag_system.clear_vector_store()
                st.success("Vector store cleared!")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat with Your Real Estate Assistant")
        
        # Display chat history
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Input area
        user_input = st.chat_input(
            "Ask about properties, listings, real estate market...",
            key="chat_input"
        )
        
        if user_input:
            # Initialize if needed
            if st.session_state.rag_system is None:
                initialize_rag()
            
            # Add user message
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Get response
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.rag_system.query(user_input)
            
            # Add assistant response
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response["answer"]
            })
            
            with st.chat_message("assistant"):
                st.markdown(response["answer"])
            
            # Show sources
            if response["sources"]:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(response["sources"], 1):
                        st.markdown(f"**Source {i}:**")
                        st.info(source["content"])
                        if source["metadata"]:
                            st.caption(f"Metadata: {source['metadata']}")
    
    with col2:
        st.subheader("🔍 Quick Actions")
        
        # Property search
        if st.button("🏡 Find Affordable Properties"):
            question = "What are the most affordable properties available?"
            user_input = question
            
            if st.session_state.rag_system:
                with st.spinner("Searching..."):
                    response = st.session_state.rag_system.query(question)
                    st.info(response["answer"])
        
        if st.button("🌟 Luxury Properties"):
            question = "What luxury properties do you have?"
            
            if st.session_state.rag_system:
                with st.spinner("Searching..."):
                    response = st.session_state.rag_system.query(question)
                    st.info(response["answer"])
        
        if st.button("📍 Location Guide"):
            question = "Tell me about the best locations and neighborhoods."
            
            if st.session_state.rag_system:
                with st.spinner("Searching..."):
                    response = st.session_state.rag_system.query(question)
                    st.info(response["answer"])
        
        st.divider()
        
        # Advanced search
        st.subheader("⚡ Advanced Search")
        
        budget = st.slider("Budget Range ($)", 50000, 1000000, (200000, 500000))
        property_type = st.selectbox(
            "Property Type",
            ["Any", "House", "Apartment", "Commercial", "Land"]
        )
        location = st.text_input("Location/Area")
        
        if st.button("🔎 Search Properties"):
            criteria = {
                "budget": f"${budget[0]:,} - ${budget[1]:,}",
                "type": property_type,
                "location": location if location else "Any"
            }
            
            if st.session_state.rag_system:
                with st.spinner("Searching..."):
                    recommendations = st.session_state.rag_system.get_property_recommendations(criteria)
                    st.success(recommendations)


if __name__ == "__main__":
    main()
