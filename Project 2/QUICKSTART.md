# 🚀 Quick Start Guide - Real Estate RAG Assistant

## ⚡ 5-Minute Setup

### Step 1: Install Ollama (2 min)
1. Download from [ollama.ai](https://ollama.ai)
2. Install and open the application

### Step 2: Pull Required Models (2 min)
Open Terminal and run:
```bash
ollama pull mistral
ollama pull nomic-embed-text
```

### Step 3: Start the Application (1 min)
```bash
# Navigate to project directory
cd "product 2"

# Install Python dependencies (first time only)
pip install -r requirements.txt

# Start Ollama (keep this running)
ollama serve

# In another terminal, start the app
streamlit run app.py
```

The app opens at: **http://localhost:8501**

---

## 🎯 First Time Usage

### 1. Initialize the System
- Click **"Initialize System"** button in the sidebar
- Wait for the RAG system to start

### 2. Load Sample Data
```bash
python utils.py sample
```
Then upload `sample_properties.csv` via the app

### 3. Test with Sample Data
```bash
python utils.py test
```

### 4. Try a Query
Ask: *"What properties are available under $500,000?"*

---

## 📁 Project Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit web interface (main app) |
| `rag_backend.py` | RAG system with LangChain |
| `document_loader.py` | Document processing |
| `utils.py` | Helper scripts and testing |
| `requirements.txt` | Python dependencies |
| `RAG.py` | Entry point script |

---

## 🔄 Typical Workflow

1. **Open Terminal 1**: Start Ollama
   ```bash
   ollama serve
   ```

2. **Open Terminal 2**: Start Streamlit App
   ```bash
   cd "product 2"
   streamlit run app.py
   ```

3. **In Browser**: Open http://localhost:8501

4. **Load Data**:
   - Upload PDFs of properties/guides
   - Upload CSV with listings
   - Add web content from real estate sites

5. **Ask Questions**:
   - Chat interface for natural language queries
   - Use quick action buttons
   - Advanced search filters

---

## ✨ Key Features

### Chat Interface
- Natural language questions
- View source documents
- Chat history maintained

### Document Management
- Upload multiple PDFs
- Import CSV listings
- Scrape web content
- Automatic text processing

### Search Features
- Budget filtering
- Property type selection
- Location filtering
- Semantic search

---

## 🆘 Troubleshooting

### Ollama won't start
```bash
# Check if port 11434 is in use
lsof -i :11434

# Or try different setup
brew services restart ollama
```

### Models not found
```bash
# List available models
ollama list

# Pull missing models
ollama pull mistral
ollama pull nomic-embed-text
```

### App won't connect
- Verify Ollama is running: `ollama serve`
- Check Python dependencies: `pip install -r requirements.txt`
- Try restarting both Ollama and Streamlit

### Slow responses
- Reduce document chunk size in `document_loader.py`
- Use faster model: `ollama pull neural-chat`
- Close other applications

---

## 💡 Tips & Tricks

### Create Better Results
- Add diverse real estate documents (guides, listings, market reports)
- Use clear, specific questions
- Ask follow-up questions for refinement

### Optimize Performance
- Start with fewer documents
- Use compression for faster retrieval
- Adjust chunk sizes for balance

### Customize Setup
- Edit model choice in sidebar
- Modify prompt templates in `rag_backend.py`
- Add custom document loaders

---

## 📚 Sample Queries

```
"Show me 3-bedroom houses under $500k"
"What are the luxury properties in this area?"
"Compare apartment vs house investments"
"Tell me about market trends for 2024"
"What should I know before buying a property?"
"Find properties with pool and garage"
```

---

## 🎓 Learning Resources

- [LangChain Documentation](https://python.langchain.com)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Ollama Models](https://ollama.ai/library)
- [RAG Concepts](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)

---

## 🚀 Next Level

After getting comfortable:
1. Add more document sources
2. Fine-tune prompts for better answers
3. Create property recommendation logic
4. Build custom integrations
5. Deploy to production (Streamlit Cloud)

---

**Ready to go? Start with: `streamlit run app.py`** 🎉
