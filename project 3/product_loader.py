# Product loader for E-Commerce Chatbot
# Loads products from CSV into MySQL and generates embeddings in ChromaDB
import pandas as pd
import logging
from pathlib import Path
from database import DatabaseManager
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from config import SAMPLE_DATA_PATH, CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, OPENAI_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductLoader:
    """Load products from CSV and initialize vector database"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        self.chroma_persist_dir = Path(CHROMA_PERSIST_DIR)
        self.chroma_persist_dir.mkdir(exist_ok=True)
    
    def load_csv(self, csv_path):
        """Load products from CSV file"""
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} products from {csv_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return None
    
    def save_products_to_mysql(self, df):
        """Save products from DataFrame to MySQL"""
        saved_products = []
        
        for idx, row in df.iterrows():
            try:
                product_id = self.db_manager.add_product(
                    name=row['name'],
                    description=row['description'],
                    price=float(row['price']),
                    category=row['category'],
                    stock=int(row['stock']),
                    sku=row.get('sku', f"SKU-{idx}"),
                    image_url=row.get('image_url', None)
                )
                
                if product_id:
                    saved_products.append({
                        'product_id': product_id,
                        'name': row['name'],
                        'description': row['description'],
                        'price': row['price'],
                        'category': row['category']
                    })
                    logger.info(f"Saved product: {row['name']} (ID: {product_id})")
            except Exception as e:
                logger.error(f"Error saving product {row['name']}: {e}")
        
        logger.info(f"Successfully saved {len(saved_products)} products to MySQL")
        return saved_products
    
    def create_product_documents(self, df):
        """Create document format for ChromaDB"""
        documents = []
        ids = []
        metadatas = []
        
        for idx, row in df.iterrows():
            # Create comprehensive product document
            doc_content = f"""
            Product: {row['name']}
            Category: {row['category']}
            Price: ${row['price']}
            Stock: {row['stock']} units available
            Description: {row['description']}
            """
            
            documents.append(doc_content)
            ids.append(f"product_{idx}")
            metadatas.append({
                'product_id': int(idx),
                'name': row['name'],
                'price': float(row['price']),
                'category': row['category'],
                'stock': int(row['stock']),
                'sku': row.get('sku', f"SKU-{idx}")
            })
        
        return documents, ids, metadatas
    
    def initialize_vector_db(self, documents, ids, metadatas):
        """Initialize ChromaDB with product embeddings"""
        try:
            # Create Chroma vector store with OpenAI embeddings
            vector_store = Chroma.from_texts(
                texts=documents,
                embedding=self.embeddings,
                ids=ids,
                metadatas=metadatas,
                collection_name=CHROMA_COLLECTION_NAME,
                persist_directory=str(self.chroma_persist_dir)
            )
            
            logger.info(f"ChromaDB initialized with {len(documents)} products")
            logger.info(f"Persisted to {self.chroma_persist_dir}")
            return vector_store
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            return None
    
    def load_products(self, csv_path=SAMPLE_DATA_PATH):
        """Main method to load all products"""
        logger.info("Starting product loading process...")
        
        # Load CSV
        df = self.load_csv(csv_path)
        if df is None:
            return False
        
        # Save to MySQL
        saved_products = self.save_products_to_mysql(df)
        if not saved_products:
            logger.warning("No products were saved to MySQL")
        
        # Create documents for vector DB
        documents, ids, metadatas = self.create_product_documents(df)
        
        # Initialize ChromaDB
        vector_store = self.initialize_vector_db(documents, ids, metadatas)
        if vector_store is None:
            return False
        
        logger.info("✓ Product loading completed successfully")
        return True


def main():
    """Run product loader"""
    logger.info("=" * 50)
    logger.info("E-Commerce Chatbot Product Loader")
    logger.info("=" * 50)
    
    loader = ProductLoader()
    success = loader.load_products()
    
    if success:
        logger.info("\n✓ All products loaded successfully!")
        logger.info("You can now start the chatbot with: streamlit run app.py")
    else:
        logger.error("\n✗ Product loading failed. Please check the errors above.")


if __name__ == "__main__":
    main()
