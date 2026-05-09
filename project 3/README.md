# 🛒 E-Commerce Chatbot

A modern e-commerce chatbot built with Streamlit, LangChain, OpenAI GPT-4, and MySQL. Features AI-powered product recommendations, shopping cart management, and order tracking.

## 🎯 Features

- **AI-Powered Chat**: Ask questions about products and get intelligent recommendations using OpenAI GPT-4
- **Product Search**: Semantic search using ChromaDB vector database for accurate product retrieval
- **Shopping Cart**: Add/remove products, view cart, and checkout
- **Order Management**: Track orders and view order history
- **Chat History**: Persistent conversation history stored in MySQL
- **Multi-Tab Interface**: Chat, Products, Cart, and Orders tabs for seamless navigation

## 🏗️ Architecture

```
Frontend (Streamlit)
    ↓
Backend (LangChain Chain)
    ↓
├── Vector Search (ChromaDB)
├── LLM (OpenAI GPT-4)
└── Database (MySQL)
```

**Key Components:**
- `app.py` - Streamlit UI with chat, products, cart, and orders
- `chatbot_backend.py` - LangChain RAG chain for conversational AI
- `database.py` - MySQL CRUD operations for users, products, orders
- `product_loader.py` - Load products from CSV into MySQL and ChromaDB
- `config.py` - Configuration management
- `init_db.sql` - Database schema

## 📋 Prerequisites

- Python 3.8+
- MySQL Server (running locally or remote)
- OpenAI API Key (GPT-4 access)
- 2GB+ free disk space (for ChromaDB)

## 🚀 Quick Start

### 1️⃣ Clone and Setup

```bash
# Navigate to project directory
cd "project 3"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Configure Environment

```bash
# Copy the template
cp .env.template .env

# Edit .env with your credentials
nano .env
```

**Required settings in `.env`:**
```
OPENAI_API_KEY=sk-your_key_here
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=ecommerce_chatbot
```

### 3️⃣ Setup MySQL Database

```bash
# Option A: Command line
mysql -u root -p ecommerce_chatbot < init_db.sql

# Option B: MySQL Workbench or GUI client
# 1. Create database: CREATE DATABASE ecommerce_chatbot;
# 2. Use: USE ecommerce_chatbot;
# 3. Run the SQL from init_db.sql
```

### 4️⃣ Load Products

```bash
# This loads sample products from CSV into MySQL and ChromaDB
python product_loader.py
```

**Expected output:**
```
Loaded 20 products from ./sample_products.csv
Successfully saved 20 products to MySQL
ChromaDB initialized with 20 products
✓ Product loading completed successfully
```

### 5️⃣ Run the Chatbot

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 💬 Usage

### Chat Tab
1. Ask questions about products: *"What wireless headphones do you have?"*
2. Get recommendations: *"Show me the best keyboards"*
3. Check stock: *"Do you have standing desks in stock?"*

### Products Tab
- Browse all products with search and category filters
- Click "Add to Cart" to add items

### Cart Tab
- Review cart items with quantity and total price
- Remove items or clear cart
- Click "Checkout" to create an order

### Orders Tab
- View order history
- Expand orders to see items and pricing

## 📂 Project Structure

```
project 3/
├── app.py                    # Streamlit UI
├── chatbot_backend.py        # LangChain chain + RAG logic
├── database.py               # MySQL CRUD operations
├── product_loader.py         # Load products into DB
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── .env.template             # Environment variables template
├── .env                      # Your credentials (create this)
├── init_db.sql               # Database schema
├── sample_products.csv       # Sample product data
├── chroma_db/                # ChromaDB persistence (created on first run)
└── README.md                 # This file
```

## 🔧 Configuration

### config.py Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `OPENAI_MODEL` | gpt-4-turbo | OpenAI model to use |
| `OPENAI_TEMPERATURE` | 0.7 | Response creativity (0=factual, 1=creative) |
| `CHROMA_COLLECTION_NAME` | products | Vector DB collection name |
| `MAX_CHAT_HISTORY` | 10 | Messages kept in context |
| `CHUNK_SIZE` | 500 | Product text chunk size |

## 🗄️ Database Schema

### Tables
- **users** - User sessions and profiles
- **products** - Product catalog with inventory
- **chat_history** - Chat messages and responses
- **shopping_cart** - Current cart items per user
- **orders** - Order records
- **order_items** - Items within orders

## 🆘 Troubleshooting

### MySQL Connection Error
```
Error: 2003 - Can't connect to MySQL server
```
**Solution:**
- Ensure MySQL server is running: `mysql.server start` (macOS) or start via Services (Windows)
- Verify credentials in `.env`
- Check port: `MYSQL_PORT=3306` (default)

### OpenAI API Error
```
AuthenticationError: Incorrect API key
```
**Solution:**
- Get API key from https://platform.openai.com/api-keys
- Ensure key starts with `sk-`
- Verify in `.env`: `OPENAI_API_KEY=sk-your_key`

### ChromaDB Not Found
```
ValueError: Could not find a collection named 'products'
```
**Solution:**
- Run `python product_loader.py` to initialize ChromaDB
- Ensure `./chroma_db` directory exists

### Streamlit Port Already in Use
```
Error: Address already in use
```
**Solution:**
```bash
streamlit run app.py --server.port 8502
```

## 📊 Sample Data

The project includes 20 sample products across 3 categories:
- **Electronics**: Headphones, Monitor, Keyboard, Webcam, Cooling Pad
- **Office**: Desk Lamp, Keyboard Wrist Rest, Standing Desk, Monitor Arm, Desk Organizer
- **Accessories**: USB-C Cable, Phone Stand, Mouse Pad, HDMI Cable, USB Hub, Docking Station, Cable Management

To add your own products, edit `sample_products.csv` and run:
```bash
python product_loader.py
```

## 🚢 Deployment

### Streamlit Cloud
```bash
# Push to GitHub
git add .
git commit -m "E-commerce chatbot"
git push

# Deploy on streamlit.io
# https://streamlit.io/cloud
```

### Docker
```dockerfile
FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

```bash
docker build -t ecommerce-chatbot .
docker run -p 8501:8501 ecommerce-chatbot
```

## 🔐 Security Notes

- Never commit `.env` file (add to `.gitignore`)
- Store OpenAI API key securely
- Validate user input in production
- Use HTTPS for production deployments
- Implement authentication for user accounts

## 📝 Future Enhancements

- [ ] User authentication (login/signup)
- [ ] Payment gateway integration (Stripe, PayPal)
- [ ] Product reviews and ratings
- [ ] Recommendation engine (collaborative filtering)
- [ ] Email notifications
- [ ] Admin dashboard
- [ ] Multi-language support
- [ ] Image upload for products

## 📚 References

- [Streamlit Docs](https://docs.streamlit.io)
- [LangChain Docs](https://python.langchain.com)
- [ChromaDB Docs](https://docs.trychroma.com)
- [OpenAI API Docs](https://platform.openai.com/docs)
- [MySQL Docs](https://dev.mysql.com/doc)

## 📧 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review logs in terminal
3. Check environment variables in `.env`
4. Verify database connectivity

## 📄 License

MIT License - Feel free to use for personal or commercial projects

---

**Built with ❤️ using Streamlit, LangChain, and OpenAI**
