# 🏠 Real Estate RAG Assistant

A powerful real estate assistant built with LangChain, Streamlit, and Ollama. This project implements Retrieval-Augmented Generation (RAG) to answer questions about properties, listings, and real estate market information.

## 🎯 Features

- **Multi-source Document Loading**: Support for PDFs, CSV listings, and web content
- **Intelligent RAG**: Uses LangChain with Ollama for context-aware responses
- **Vector Database**: Chroma for efficient document retrieval
- **Streamlit UI**: Interactive web interface for querying
- **Real Estate Specialized**: Handles property listings, guides, contracts, and market data
- **Source Attribution**: Shows relevant documents for each answer
- **Advanced Search**: Property filtering by budget, type, and location

## 📋 Prerequisites

- **Ollama**: Download and install from [ollama.ai](https://ollama.ai)
- **Python 3.8+**: For running the application
- **Required Models**: 
  - `mistral` (for LLM): `ollama pull mistral`
  - `nomic-embed-text` (for embeddings): `ollama pull nomic-embed-text`

## 🚀 Installation

1. **Clone/Navigate to project directory**:
```bash
cd "product 2"
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Start Ollama** (in a separate terminal):
```bash
ollama serve
```

4. **Pull required models** (if not already done):
```bash
ollama pull mistral
ollama pull nomic-embed-text
```

## 🎮 Usage

### Start the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Workflow

1. **Initialize System**: Click "Initialize System" button in sidebar
2. **Load Documents**:
   - **Upload PDFs**: Upload property brochures, contracts, or real estate guides
   - **Load Listings**: Upload CSV file with property listings
   - **Add Web Content**: Paste URLs of real estate websites or articles
3. **Query**: Ask questions about properties, market trends, or recommendations
4. **Review Sources**: Check document sources for each answer

## 📁 Project Structure

```
product 2/
├── app.py                 # Streamlit frontend
├── rag_backend.py        # LangChain RAG system
├── document_loader.py    # Document processing
├── requirements.txt      # Dependencies
├── README.md            # This file
└── chroma_db/           # Vector database (created on first run)
```

## 🔧 Module Details

### `app.py` - Streamlit Frontend
- Interactive chat interface
- Document upload and management
- Advanced property search filters
- Quick action buttons
- Source document viewing

### `rag_backend.py` - RAG System
- Vector store management with Chroma
- LLM and embedding initialization
- Query processing with retrieval
- Property recommendation system
- Context compression for better results

### `document_loader.py` - Document Processing
- PDF loading and text extraction
- CSV property listing parsing
- Web scraping and cleaning
- Text chunking and splitting
- Metadata preservation

## 📊 Sample Data

### Create Sample CSV Listings

Create `properties.csv`:
```csv
Address,Type,Price,Bedrooms,Bathrooms,Area,Features
123 Main St,House,450000,3,2,2000,"Garden, Garage"
456 Oak Ave,Apartment,350000,2,1.5,1200,"Balcony, Pool"
789 Pine Rd,House,550000,4,3,3000,"Patio, Pool, Gym"
```

Then upload via the app's "Load Listings" tab.

### Sample PDF Content

Upload any real estate:
- Property brochures
- Market analysis reports
- Investment guides
- Neighborhood information
- Contract templates

### Web Content

Paste URLs from:
- Real estate websites
- Property listing portals
- Market analysis sites
- Neighborhood guides

## 🎓 Example Queries

- "What properties are available under $500,000?"
- "Tell me about luxury homes in this area"
- "What are the best neighborhoods?"
- "Can you compare these two properties?"
- "What's the market trend for this area?"
- "What should I know about buying investment properties?"

## ⚙️ Configuration

### Change Ollama Model

Edit the model in sidebar or directly in code:

```python
st.session_state.rag_system = RealEstateRAG(
    model_name="neural-chat",  # Change this
    embedding_model="nomic-embed-text"
)
```

**Available Models** (use `ollama pull <model>`):
- `mistral` - Recommended, good balance
- `neural-chat` - Good conversational ability
- `llama2` - Large, more capable
- `orca-mini` - Smaller, faster

### Adjust Retrieval Parameters

In `rag_backend.py`:
```python
base_retriever = self.vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}  # Adjust k for more/fewer results
)
```

## 🔄 Troubleshooting

### "Connection refused" Error
- Make sure Ollama is running: `ollama serve`
- Check that it's on `http://localhost:11434` (default)

### No documents loaded error
- Click "Initialize System" first
- Then upload documents in the sidebar
- Wait for processing to complete

### Slow responses
- Reduce `k` parameter in retriever (faster but fewer results)
- Use `neural-chat` or `orca-mini` instead of `mistral`
- Close other applications to free up memory

### "Model not found" Error
- Pull the required model: `ollama pull mistral`
- Check available models: `ollama list`

## 📈 Performance Tips

1. **Chunk Size**: Adjust in `document_loader.py`:
   ```python
   RecursiveCharacterTextSplitter(chunk_size=1000)
   ```

2. **Vector Store**: Use compression to improve retrieval speed
   - Set `fetch_k` higher than `k` for better results

3. **Batch Processing**: Load multiple documents at once

## 🛠️ Development

### Add New Document Types

In `document_loader.py`:
```python
def load_new_format(self, file_path: str) -> List[Document]:
    # Your loading logic
    documents = []
    # Process file
    return self.text_splitter.split_documents(documents)
```

### Customize Prompts

In `rag_backend.py`:
```python
prompt_template = ChatPromptTemplate.from_template(
    """Your custom prompt here..."""
)
```

### Add New Query Types

Create methods in `RealEstateRAG` class:
```python
def specialized_query(self, params: dict) -> str:
    # Your custom logic
    return self.query(formatted_question)
```

## 🤝 Contributing

To improve this project:
1. Optimize document loading for specific formats
2. Add more real estate-specific query templates
3. Implement caching for faster responses
4. Add API endpoints for integration
5. Create sample datasets

## 📝 License

Open source - use and modify as needed.

## 🚀 Next Steps

1. **Production Deployment**:
   - Use `streamlit deploy` for cloud hosting
   - Set up proper Ollama serving
   - Add authentication

2. **Advanced Features**:
   - Multi-language support
   - Property image analysis
   - Market prediction
   - Integration with MLS databases

3. **Optimization**:
   - Fine-tune models for real estate domain
   - Implement caching layer
   - Add query logging and analytics

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section
2. Verify Ollama is running
3. Ensure all dependencies are installed
4. Check available models: `ollama list`

---

**Built with ❤️ using LangChain, Streamlit, and Ollama**
