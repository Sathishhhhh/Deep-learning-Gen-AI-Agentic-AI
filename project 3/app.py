# E-Commerce Chatbot Streamlit App
import streamlit as st
import uuid
import logging
from datetime import datetime
from chatbot_backend import initialize_chatbot
from database import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="E-Commerce Chatbot",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { padding: 20px; }
    .chat-message { 
        padding: 10px; 
        margin: 5px 0; 
        border-radius: 5px;
    }
    .user-message {
        background-color: #e3f2fd;
        text-align: right;
    }
    .bot-message {
        background-color: #f5f5f5;
    }
    .product-card {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.user_id = None
    st.session_state.chat_history = []
    st.session_state.chatbot = None
    st.session_state.db_manager = None

# Initialize database
if st.session_state.db_manager is None:
    st.session_state.db_manager = DatabaseManager()

# Get or create user
if st.session_state.user_id is None:
    user_id = st.session_state.db_manager.get_user_by_session(st.session_state.session_id)
    if user_id is None:
        user_id = st.session_state.db_manager.create_user(st.session_state.session_id)
    st.session_state.user_id = user_id

# Initialize chatbot
if st.session_state.chatbot is None:
    with st.spinner("Initializing chatbot..."):
        st.session_state.chatbot = initialize_chatbot()

# Load chat history on first run
if not st.session_state.chat_history and st.session_state.user_id:
    st.session_state.chat_history = st.session_state.db_manager.get_chat_history(
        st.session_state.user_id, 
        limit=20
    )


# Sidebar
with st.sidebar:
    st.title("🛒 E-Commerce Chatbot")
    st.markdown("---")
    
    # User info
    st.subheader("👤 User Info")
    st.write(f"Session ID: `{st.session_state.session_id[:8]}...`")
    
    # Quick actions
    st.subheader("⚡ Quick Actions")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 New Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear History"):
            st.session_state.db_manager.clear_cart(st.session_state.user_id)
            st.session_state.chat_history = []
            st.rerun()
    
    # Navigation
    st.markdown("---")
    st.subheader("📚 Navigation")
    page = st.radio(
        "Select Page:",
        ["💬 Chat", "🛍️ Products", "🛒 Cart", "📋 Orders"]
    )


# Main content based on page
if page == "💬 Chat":
    st.title("💬 Chat with Our Assistant")
    st.markdown("Ask questions about our products, get recommendations, or manage your shopping.")
    
    # Chat history display
    st.subheader("Chat History")
    
    if st.session_state.chatbot:
        chat_container = st.container()
        
        # Display existing chat history
        with chat_container:
            for msg in st.session_state.chat_history:
                user_msg, bot_msg, timestamp = msg
                
                st.markdown(f'<div class="chat-message user-message">👤: {user_msg}</div>', 
                           unsafe_allow_html=True)
                st.markdown(f'<div class="chat-message bot-message">🤖: {bot_msg}</div>', 
                           unsafe_allow_html=True)
                st.caption(f"_{timestamp}_")
        
        # User input
        st.markdown("---")
        user_input = st.text_input("You:", placeholder="Ask me about products, prices, or recommendations...")
        
        if user_input:
            # Show user message
            st.markdown(f'<div class="chat-message user-message">👤: {user_input}</div>', 
                       unsafe_allow_html=True)
            
            # Get bot response
            with st.spinner("Thinking..."):
                bot_response = st.session_state.chatbot.chat(user_input, st.session_state.user_id)
            
            # Show bot response
            st.markdown(f'<div class="chat-message bot-message">🤖: {bot_response}</div>', 
                       unsafe_allow_html=True)
            
            # Add to chat history
            st.session_state.chat_history.append((user_input, bot_response, datetime.now().strftime("%H:%M:%S")))
            
            st.rerun()
    else:
        st.error("Failed to initialize chatbot. Please check your configuration.")


elif page == "🛍️ Products":
    st.title("🛍️ Browse Products")
    
    db_manager = st.session_state.db_manager
    
    # Search and filter
    col1, col2 = st.columns(2)
    
    with col1:
        search_query = st.text_input("Search products:", placeholder="e.g., headphones, keyboard")
    
    with col2:
        category_filter = st.selectbox(
            "Filter by category:",
            ["All", "Electronics", "Office", "Accessories"]
        )
    
    # Get products
    all_products = db_manager.get_all_products()
    
    # Filter and search
    filtered_products = []
    for product in all_products:
        product_id, name, description, price, category, stock = product
        
        if category_filter != "All" and category != category_filter:
            continue
        
        if search_query and search_query.lower() not in name.lower() and search_query.lower() not in description.lower():
            continue
        
        filtered_products.append(product)
    
    # Display products
    st.subheader(f"Found {len(filtered_products)} products")
    
    cols = st.columns(3)
    for idx, product in enumerate(filtered_products):
        product_id, name, description, price, category, stock = product
        
        col = cols[idx % 3]
        with col:
            st.markdown(f'<div class="product-card">', unsafe_allow_html=True)
            st.markdown(f"### {name}")
            st.write(f"**Price:** ${price}")
            st.write(f"**Category:** {category}")
            st.write(f"**Stock:** {stock} units")
            st.write(f"_{description[:100]}..._")
            
            if st.button(f"Add to Cart", key=f"add_{product_id}"):
                db_manager.add_to_cart(st.session_state.user_id, product_id, quantity=1)
                st.success(f"Added {name} to cart!")
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)


elif page == "🛒 Cart":
    st.title("🛒 Shopping Cart")
    
    db_manager = st.session_state.db_manager
    cart_items = db_manager.get_cart(st.session_state.user_id)
    
    if not cart_items:
        st.info("Your cart is empty. Browse products to add items!")
    else:
        st.subheader(f"Items in Cart: {len(cart_items)}")
        
        # Display cart items
        total = 0
        for item in cart_items:
            cart_id, product_id, name, price, quantity, item_total = item
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.write(name)
            with col2:
                st.write(f"${price}")
            with col3:
                st.write(quantity)
            with col4:
                st.write(f"${item_total:.2f}")
            with col5:
                if st.button("❌", key=f"remove_{product_id}"):
                    db_manager.remove_from_cart(st.session_state.user_id, product_id)
                    st.rerun()
            
            total += item_total
        
        # Cart total
        st.markdown("---")
        st.subheader(f"Total: ${total:.2f}")
        
        # Checkout button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Checkout"):
                order_id = db_manager.create_order(st.session_state.user_id, cart_items)
                if order_id:
                    st.success(f"✓ Order #{order_id} created successfully!")
                    st.info("Thank you for your purchase!")
                    st.rerun()
                else:
                    st.error("Failed to create order. Please try again.")
        
        with col2:
            if st.button("🗑️ Clear Cart"):
                db_manager.clear_cart(st.session_state.user_id)
                st.rerun()


elif page == "📋 Orders":
    st.title("📋 Order History")
    
    db_manager = st.session_state.db_manager
    orders = db_manager.get_user_orders(st.session_state.user_id)
    
    if not orders:
        st.info("You haven't placed any orders yet.")
    else:
        st.subheader(f"Your Orders: {len(orders)}")
        
        for order in orders:
            order_id, total_price, status, created_at = order
            
            with st.expander(f"Order #{order_id} - ${total_price:.2f} ({status.upper()}) - {created_at}"):
                # Get order details
                order_items = db_manager.get_order_details(order_id)
                
                if order_items:
                    st.write("**Items:**")
                    for item in order_items:
                        product_id, name, quantity, price, total = item
                        st.write(f"- {name}: {quantity}x ${price} = ${total:.2f}")
                else:
                    st.write("No items in this order.")


# Footer
st.markdown("---")
st.caption("💡 Powered by LangChain + OpenAI + ChromaDB | MySQL Database")
