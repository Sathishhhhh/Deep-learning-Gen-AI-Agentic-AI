"""
Document loader for real estate content (PDFs, web pages, and property listings)
"""
import os
from typing import List
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


class RealEstateDocumentLoader:
    """Load and process real estate documents from multiple sources"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """Load a PDF document"""
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            print(f"Error loading PDF {pdf_path}: {e}")
            return []
    
    def load_pdfs_from_folder(self, folder_path: str) -> List[Document]:
        """Load all PDFs from a folder"""
        documents = []
        pdf_folder = Path(folder_path)
        
        for pdf_file in pdf_folder.glob("*.pdf"):
            print(f"Loading {pdf_file}...")
            docs = self.load_pdf(str(pdf_file))
            documents.extend(docs)
        
        return documents
    
    def load_property_listing(self, listing_text: str, metadata: dict = None) -> List[Document]:
        """Load property listing text"""
        if metadata is None:
            metadata = {}
        
        doc = Document(page_content=listing_text, metadata=metadata)
        return self.text_splitter.split_documents([doc])
    
    def scrape_web_content(self, url: str) -> List[Document]:
        """Scrape and process web content"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            metadata = {"source": url, "type": "web"}
            doc = Document(page_content=text, metadata=metadata)
            
            return self.text_splitter.split_documents([doc])
        
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return []
    
    def load_csv_listings(self, csv_path: str) -> List[Document]:
        """Load property listings from CSV file"""
        try:
            import pandas as pd
            
            df = pd.read_csv(csv_path)
            documents = []
            
            for _, row in df.iterrows():
                # Create a text representation of the listing
                listing_text = "\n".join([
                    f"{col}: {val}" for col, val in row.items() if pd.notna(val)
                ])
                
                metadata = {
                    "source": csv_path,
                    "type": "property_listing",
                    **row.to_dict()
                }
                
                doc = Document(page_content=listing_text, metadata=metadata)
                documents.extend(self.text_splitter.split_documents([doc]))
            
            return documents
        
        except Exception as e:
            print(f"Error loading CSV {csv_path}: {e}")
            return []
