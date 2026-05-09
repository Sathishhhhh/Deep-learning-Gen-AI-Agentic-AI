# Chatbot backend for E-Commerce Chatbot
# Uses LangChain with OpenAI and ChromaDB for product retrieval
import logging
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from config import (
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE,
    CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
)
from database import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EcommerceChatbot:
    """E-Commerce Chatbot using LangChain and ChromaDB"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=OPENAI_MODEL,
            temperature=OPENAI_TEMPERATURE
        )
        
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db_manager = DatabaseManager()
        
        # Load ChromaDB vector store
        self.vector_store = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_PERSIST_DIR
        )
        
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3}
        )
        
        # Define the RAG prompt
        self.prompt = ChatPromptTemplate.from_template("""
You are a helpful e-commerce customer service chatbot. You help customers find products, answer questions about inventory, and assist with their shopping.

Use the following product information to answer customer questions accurately and helpfully.

Product Context:
{context}

Customer Question: {question}

Please provide a helpful response. If the customer is asking about products, recommend the most relevant ones based on the context. 
If you don't have relevant product information, politely let them know what you can help with.

Response:
""")
        
        # Create the RAG chain
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
        )
    
    def format_retrieved_docs(self, docs):
        """Format retrieved documents for context"""
        if not docs:
            return "No relevant products found."
        
        formatted = []
        for doc in docs:
            formatted.append(f"- {doc.page_content}")
        
        return "\n".join(formatted)
    
    def chat(self, user_message, user_id=None):
        """Process user message and return bot response"""
        try:
            # Get response from chain
            response = self.chain.invoke(user_message)
            bot_response = response.content
            
            # Extract product information if available
            product_id = self._extract_product_id(user_message)
            
            # Save chat history
            if user_id:
                self.db_manager.save_chat_message(
                    user_id=user_id,
                    user_message=user_message,
                    bot_response=bot_response,
                    product_id=product_id
                )
            
            return bot_response
        
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return "I apologize, but I encountered an error processing your request. Please try again."
    
    def search_products(self, query, num_results=5):
        """Search for products by query"""
        try:
            results = self.vector_store.similarity_search(query, k=num_results)
            return results
        except Exception as e:
            logger.error(f"Error searching products: {e}")
            return []
    
    def get_product_recommendations(self, product_id, num_recommendations=3):
        """Get product recommendations similar to a given product"""
        try:
            product = self.db_manager.get_product_by_id(product_id)
            
            if not product:
                return []
            
            _, name, description, price, category, _ = product
            
            # Search for similar products
            query = f"{name} {category} {description}"
            results = self.vector_store.similarity_search(query, k=num_recommendations)
            
            return results
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []
    
    def _extract_product_id(self, user_message):
        """Try to extract product ID from user message"""
        # Simple extraction - can be enhanced
        import re
        match = re.search(r'product[_\s]?(\d+)', user_message.lower())
        return int(match.group(1)) if match else None
    
    def get_conversation_context(self, user_id, max_history=5):
        """Get recent chat history for context"""
        try:
            history = self.db_manager.get_chat_history(user_id, limit=max_history)
            return history
        except Exception as e:
            logger.error(f"Error retrieving conversation context: {e}")
            return []
    
    def handle_cart_query(self, user_message, user_id):
        """Handle shopping cart related queries"""
        if "cart" in user_message.lower():
            if "add" in user_message.lower():
                # Extract product info and add to cart
                return "I can help you add items to your cart. Which product would you like to add?"
            elif "view" in user_message.lower() or "show" in user_message.lower():
                # Get cart contents
                cart_items = self.db_manager.get_cart(user_id)
                if not cart_items:
                    return "Your cart is empty."
                
                cart_text = "Here's your cart:\n"
                total = 0
                for item in cart_items:
                    _, _, name, price, quantity, item_total = item
                    cart_text += f"- {name}: {quantity}x ${price} = ${item_total}\n"
                    total += item_total
                
                cart_text += f"\nTotal: ${total:.2f}"
                return cart_text
            elif "clear" in user_message.lower():
                # Clear cart
                self.db_manager.clear_cart(user_id)
                return "Your cart has been cleared."
        
        return None


def initialize_chatbot():
    """Initialize the chatbot"""
    try:
        logger.info("Initializing E-Commerce Chatbot...")
        chatbot = EcommerceChatbot()
        logger.info("✓ Chatbot initialized successfully")
        return chatbot
    except Exception as e:
        logger.error(f"Error initializing chatbot: {e}")
        return None


if __name__ == "__main__":
    chatbot = initialize_chatbot()
    if chatbot:
        # Test query
        response = chatbot.chat("What wireless headphones do you have?")
        print(f"\nTest Query Response:\n{response}")
