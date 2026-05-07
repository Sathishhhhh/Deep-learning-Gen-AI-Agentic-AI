"""
RAG Backend using LangChain and Ollama for Real Estate Assistant
"""
import os
from typing import List, Optional
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import chromadb


class RealEstateRAG:
    """Real Estate RAG System using LangChain and Ollama"""
    
    def __init__(
        self,
        model_name: str = "mistral",
        embedding_model: str = "nomic-embed-text",
        vector_db_path: str = "./chroma_db"
    ):
        """
        Initialize the RAG system
        
        Args:
            model_name: Ollama model to use (e.g., mistral, neural-chat, llama2)
            embedding_model: Embedding model to use
            vector_db_path: Path to store vector database
        """
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.vector_db_path = vector_db_path
        
        # Initialize LLM
        self.llm = OllamaLLM(model=model_name, temperature=0.3)
        
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        
        # Initialize vector store
        self.vector_store = None
        self.retriever = None
        
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store"""
        print(f"Adding {len(documents)} documents to vector store...")
        
        if self.vector_store is None:
            # Create new vector store
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.vector_db_path,
                collection_name="real_estate"
            )
        else:
            # Add to existing vector store
            self.vector_store.add_documents(documents)
        
        # Update retriever
        self._initialize_retriever()
        print("Documents added successfully!")
    
    def _initialize_retriever(self) -> None:
        """Initialize the retriever with MMR search"""
        if self.vector_store is None:
            return
        
        # Use MMR (Maximum Marginal Relevance) retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20}
        )
    
    def load_vector_store(self) -> None:
        """Load existing vector store from disk"""
        if os.path.exists(self.vector_db_path):
            print(f"Loading vector store from {self.vector_db_path}...")
            self.vector_store = Chroma(
                persist_directory=self.vector_db_path,
                embedding_function=self.embeddings,
                collection_name="real_estate"
            )
            self._initialize_retriever()
            print("Vector store loaded!")
        else:
            print(f"No vector store found at {self.vector_db_path}")
    
    def query(self, question: str, use_memory: bool = True) -> dict:
        """
        Query the real estate knowledge base
        
        Args:
            question: User question about real estate
            use_memory: Whether to use retrieval chain
        
        Returns:
            Dictionary with answer and source documents
        """
        if self.retriever is None:
            return {
                "answer": "No documents loaded. Please add documents first.",
                "sources": []
            }
        
        # Retrieve relevant documents
        docs = self.retriever.invoke(question)
        
        # Create prompt for RAG
        prompt = ChatPromptTemplate.from_template(
            """You are a helpful real estate assistant. Answer the user's question based on the provided context about properties, listings, and real estate market information.

Context:
{context}

Question: {question}

Provide a helpful, accurate answer about real estate. If you don't have enough information in the context, say so."""
        )
        
        # Format context
        context = "\n".join([doc.page_content for doc in docs])
        
        # Create chain
        chain = prompt | self.llm
        
        # Get response
        answer = chain.invoke({"context": context, "question": question})
        
        # Format sources
        sources = [
            {
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata
            }
            for doc in docs
        ]
        
        return {
            "answer": answer,
            "sources": sources
        }
    
    def real_estate_query(self, question: str) -> dict:
        """
        Specialized query for real estate questions
        """
        # Create a specialized prompt for real estate
        prompt_template = ChatPromptTemplate.from_template(
            """You are a helpful real estate assistant. Answer the user's question based on the provided context about properties, listings, and real estate market information.
            
Context: {context}

Question: {question}

Provide a helpful, accurate answer about real estate. If you don't have enough information, say so."""
        )
        
        return self.query(question)
    
    def get_property_recommendations(self, criteria: dict) -> str:
        """
        Get property recommendations based on criteria
        
        Args:
            criteria: Dictionary with search criteria (e.g., budget, location, type)
        
        Returns:
            Recommendation text
        """
        criteria_text = ", ".join([f"{k}: {v}" for k, v in criteria.items()])
        question = f"Based on these criteria: {criteria_text}, what properties would you recommend?"
        
        result = self.query(question)
        return result["answer"]
    
    def clear_vector_store(self) -> None:
        """Clear the vector store"""
        import shutil
        if os.path.exists(self.vector_db_path):
            shutil.rmtree(self.vector_db_path)
        self.vector_store = None
        self.retriever = None
        print("Vector store cleared!")
