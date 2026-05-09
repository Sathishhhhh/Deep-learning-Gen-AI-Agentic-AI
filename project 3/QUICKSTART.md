# QUICKSTART.md - E-Commerce Chatbot Setup Guide

## ⚡ 5-Minute Setup

### Step 1: Install Dependencies (1 min)
```bash
cd "project 3"
pip install -r requirements.txt
```

### Step 2: Configure Environment (1 min)
```bash
cp .env.template .env
# Edit .env with your credentials:
# - OPENAI_API_KEY: Get from https://platform.openai.com/api-keys
# - MYSQL credentials: Your MySQL server info
```

### Step 3: Setup MySQL (2 min)
```bash
# Run this in MySQL
mysql -u root -p
> CREATE DATABASE ecommerce_chatbot;
> USE ecommerce_chatbot;
> source init_db.sql;
```

### Step 4: Load Products (30 sec)
```bash
python product_loader.py
```

### Step 5: Launch App (30 sec)
```bash
streamlit run app.py
```

✅ Done! App runs at `http://localhost:8501`

---

## 📋 Files Created

| File | Purpose |
|------|---------|
| `requirements.txt` | Python dependencies (install with pip) |
| `config.py` | Configuration settings (edit for custom values) |
| `.env.template` | Template for environment variables (copy to `.env`) |
| `.env` | Your credentials (create from template, don't commit) |
| `init_db.sql` | MySQL database schema (run in MySQL) |
| `sample_products.csv` | 20 sample products (edit to add your products) |
| `database.py` | MySQL CRUD operations for all tables |
| `chatbot_backend.py` | LangChain RAG chain with OpenAI + ChromaDB |
| `product_loader.py` | Loads CSV products into MySQL + ChromaDB |
| `app.py` | Streamlit UI with 4 tabs: Chat, Products, Cart, Orders |
| `README.md` | Full documentation |

---

## 🎯 What Each Tab Does

### 💬 Chat
- Ask questions: "What keyboards do you have?"
- Get recommendations
- View chat history

### 🛍️ Products
- Search products
- Filter by category
- Add to cart

### 🛒 Cart
- View items
- Update quantity
- Checkout

### 📋 Orders
- Order history
- View order details
- Track purchases

---

## 🔑 Key Environment Variables

```
OPENAI_API_KEY=sk-...             # OpenAI API key
MYSQL_HOST=localhost              # MySQL host
MYSQL_USER=root                   # MySQL username
MYSQL_PASSWORD=...                # MySQL password
MYSQL_DATABASE=ecommerce_chatbot  # Database name
```

---

## 🚨 Common Issues & Solutions

| Problem | Solution |
|---------|----------|
| MySQL connection error | Ensure MySQL is running; check credentials in `.env` |
| OpenAI API error | Verify API key and check quota at platform.openai.com |
| ChromaDB not found | Run `python product_loader.py` first |
| Port 8501 in use | `streamlit run app.py --server.port 8502` |

---

## 📚 Architecture Overview

```
User Input (Streamlit)
        ↓
App.py (Chat, Products, Cart, Orders)
        ↓
Database.py (MySQL CRUD)  +  Chatbot_backend.py (LangChain Chain)
        ↓                            ↓
MySQL Database              ChromaDB (Vector Search) + OpenAI GPT-4
```

---

## 🚀 What's Next?

1. **Test the chat**: Try asking "What products do you have?"
2. **Browse products**: Use search to find items
3. **Add to cart**: Click "Add to Cart" on products
4. **Checkout**: Complete the order
5. **View orders**: Check order history tab

---

## 💡 Customization Tips

- **Add products**: Edit `sample_products.csv` and run `python product_loader.py`
- **Change LLM**: Edit `OPENAI_MODEL` in `config.py` (use gpt-3.5-turbo for cost savings)
- **Custom prompt**: Modify the prompt in `chatbot_backend.py`
- **UI styling**: Edit CSS in `app.py` under `st.markdown("<style>...")`

---

## 📖 Full Documentation

See `README.md` for:
- Detailed setup instructions
- Deployment guides (Docker, Streamlit Cloud)
- Database schema details
- API references
- Troubleshooting guide

---

**You're all set! 🎉 Start with `streamlit run app.py`**
