"""
Utility script for setting up and testing the Real Estate RAG Assistant
"""
import os
import sys
from document_loader import RealEstateDocumentLoader
from rag_backend import RealEstateRAG


def create_sample_data():
    """Create sample CSV data for testing"""
    import pandas as pd
    
    sample_properties = {
        "Address": [
            "123 Main Street, Downtown",
            "456 Oak Avenue, Suburbs",
            "789 Pine Road, Hillside",
            "321 Elm Street, Downtown",
            "654 Birch Lane, Suburbs",
        ],
        "Type": ["House", "Apartment", "House", "Condo", "House"],
        "Price": [450000, 350000, 550000, 425000, 480000],
        "Bedrooms": [3, 2, 4, 2, 3],
        "Bathrooms": [2, 1.5, 3, 2, 2.5],
        "SquareFeet": [2000, 1200, 3000, 1500, 2200],
        "YearBuilt": [1995, 2005, 1988, 2015, 2001],
        "Features": [
            "Garage, Backyard, Patio",
            "Pool Access, Balcony, Gym",
            "Pool, Patio, Garden, Garage",
            "Rooftop Terrace, Doorman",
            "Workshop, Large Lot, Deck",
        ],
        "Description": [
            "Beautiful family home in quiet neighborhood with large backyard perfect for entertaining",
            "Modern apartment with great amenities and close to downtown shopping and dining",
            "Spacious house on hilltop with views, ideal for families seeking luxury",
            "Trendy condo in heart of city, walk to restaurants, shops, and entertainment",
            "Charming suburban home with large property, perfect for gardening enthusiasts",
        ]
    }
    
    df = pd.DataFrame(sample_properties)
    df.to_csv("sample_properties.csv", index=False)
    print("✅ Created sample_properties.csv")
    return df


def test_rag_system():
    """Test the RAG system with sample data"""
    print("\n🧪 Testing Real Estate RAG System...\n")
    
    # Initialize
    print("1️⃣  Initializing RAG system...")
    rag = RealEstateRAG(
        model_name="mistral",
        embedding_model="nomic-embed-text",
        vector_db_path="./chroma_db"
    )
    print("   ✅ RAG system initialized\n")
    
    # Load sample data
    print("2️⃣  Loading sample properties...")
    loader = RealEstateDocumentLoader()
    
    # Create and load sample CSV
    if not os.path.exists("sample_properties.csv"):
        create_sample_data()
    
    docs = loader.load_csv_listings("sample_properties.csv")
    print(f"   ✅ Loaded {len(docs)} document chunks\n")
    
    # Add to vector store
    print("3️⃣  Adding to vector store...")
    rag.add_documents(docs)
    print("   ✅ Documents added\n")
    
    # Test queries
    print("4️⃣  Testing queries...\n")
    
    test_queries = [
        "What houses are available?",
        "Show me affordable properties",
        "What are the luxury properties?",
        "Tell me about properties in downtown area"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"   Query {i}: {query}")
        try:
            result = rag.query(query)
            print(f"   Answer: {result['answer'][:200]}...")
            print(f"   Sources found: {len(result['sources'])}\n")
        except Exception as e:
            print(f"   ❌ Error: {e}\n")


def check_ollama():
    """Check if Ollama is running and models are available"""
    import requests
    
    print("🔍 Checking Ollama setup...\n")
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = response.json()
        
        print("✅ Ollama is running!")
        print("\n📦 Available models:")
        
        if "models" in models:
            for model in models["models"]:
                print(f"   - {model['name']}")
        else:
            print("   No models found")
        
        # Check required models
        model_names = [m['name'] for m in models.get("models", [])]
        
        print("\n📋 Required models:")
        required = ["mistral", "nomic-embed-text"]
        for model in required:
            if any(model in m for m in model_names):
                print(f"   ✅ {model}")
            else:
                print(f"   ❌ {model} - Run: ollama pull {model}")
        
    except Exception as e:
        print("❌ Ollama is not running!")
        print("   Start Ollama with: ollama serve")
        print(f"   Error: {e}")


def main():
    """Main utility function"""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "check":
            check_ollama()
        elif command == "sample":
            create_sample_data()
        elif command == "test":
            check_ollama()
            print("\n" + "="*50)
            test_rag_system()
        else:
            print("Usage:")
            print("  python utils.py check   - Check Ollama status")
            print("  python utils.py sample  - Create sample data")
            print("  python utils.py test    - Run full test")
    else:
        print("Real Estate RAG - Utility Script")
        print("\nUsage:")
        print("  python utils.py check   - Check Ollama setup and models")
        print("  python utils.py sample  - Create sample property CSV")
        print("  python utils.py test    - Run full system test")
        print("\nOr start the app with:")
        print("  streamlit run app.py")


if __name__ == "__main__":
    main()
