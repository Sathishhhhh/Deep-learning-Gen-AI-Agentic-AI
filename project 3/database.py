# Database operations for E-Commerce Chatbot
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import logging
from config import MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE, MYSQL_PORT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages MySQL database operations"""
    
    def __init__(self):
        self.host = MYSQL_HOST
        self.user = MYSQL_USER
        self.password = MYSQL_PASSWORD
        self.database = MYSQL_DATABASE
        self.port = MYSQL_PORT
    
    def get_connection(self):
        """Get MySQL database connection"""
        try:
            connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                port=self.port
            )
            return connection
        except Error as e:
            logger.error(f"Database connection error: {e}")
            return None
    
    # ============ User Management ============
    def create_user(self, session_id, username=None, email=None):
        """Create a new user session"""
        conn = self.get_connection()
        if not conn:
            return None
        
        try:
            cursor = conn.cursor()
            query = "INSERT INTO users (session_id, username, email) VALUES (%s, %s, %s)"
            cursor.execute(query, (session_id, username, email))
            conn.commit()
            user_id = cursor.lastrowid
            cursor.close()
            return user_id
        except Error as e:
            logger.error(f"Error creating user: {e}")
            return None
        finally:
            conn.close()
    
    def get_user_by_session(self, session_id):
        """Get user ID by session ID"""
        conn = self.get_connection()
        if not conn:
            return None
        
        try:
            cursor = conn.cursor()
            query = "SELECT user_id FROM users WHERE session_id = %s"
            cursor.execute(query, (session_id,))
            result = cursor.fetchone()
            cursor.close()
            return result[0] if result else None
        except Error as e:
            logger.error(f"Error retrieving user: {e}")
            return None
        finally:
            conn.close()
    
    # ============ Chat History ============
    def save_chat_message(self, user_id, user_message, bot_response, product_id=None):
        """Save chat message to database"""
        conn = self.get_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            query = "INSERT INTO chat_history (user_id, user_message, bot_response, product_id) VALUES (%s, %s, %s, %s)"
            cursor.execute(query, (user_id, user_message, bot_response, product_id))
            conn.commit()
            cursor.close()
            return True
        except Error as e:
            logger.error(f"Error saving chat message: {e}")
            return False
        finally:
            conn.close()
    
    def get_chat_history(self, user_id, limit=10):
        """Get chat history for a user"""
        conn = self.get_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            query = """
                SELECT user_message, bot_response, timestamp 
                FROM chat_history 
                WHERE user_id = %s 
                ORDER BY timestamp DESC 
                LIMIT %s
            """
            cursor.execute(query, (user_id, limit))
            results = cursor.fetchall()
            cursor.close()
            return list(reversed(results))
        except Error as e:
            logger.error(f"Error retrieving chat history: {e}")
            return []
        finally:
            conn.close()
    
    # ============ Products ============
    def add_product(self, name, description, price, category, stock, sku, image_url=None):
        """Add a new product"""
        conn = self.get_connection()
        if not conn:
            return None
        
        try:
            cursor = conn.cursor()
            query = """
                INSERT INTO products (name, description, price, category, stock, sku, image_url) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (name, description, price, category, stock, sku, image_url))
            conn.commit()
            product_id = cursor.lastrowid
            cursor.close()
            return product_id
        except Error as e:
            logger.error(f"Error adding product: {e}")
            return None
        finally:
            conn.close()
    
    def get_all_products(self):
        """Get all products"""
        conn = self.get_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            query = "SELECT product_id, name, description, price, category, stock FROM products"
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            return results
        except Error as e:
            logger.error(f"Error retrieving products: {e}")
            return []
        finally:
            conn.close()
    
    def get_product_by_id(self, product_id):
        """Get product by ID"""
        conn = self.get_connection()
        if not conn:
            return None
        
        try:
            cursor = conn.cursor()
            query = "SELECT product_id, name, description, price, category, stock FROM products WHERE product_id = %s"
            cursor.execute(query, (product_id,))
            result = cursor.fetchone()
            cursor.close()
            return result
        except Error as e:
            logger.error(f"Error retrieving product: {e}")
            return None
        finally:
            conn.close()
    
    # ============ Shopping Cart ============
    def add_to_cart(self, user_id, product_id, quantity=1):
        """Add item to cart"""
        conn = self.get_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            # Check if item already in cart
            query = "SELECT quantity FROM shopping_cart WHERE user_id = %s AND product_id = %s"
            cursor.execute(query, (user_id, product_id))
            result = cursor.fetchone()
            
            if result:
                # Update quantity
                new_quantity = result[0] + quantity
                update_query = "UPDATE shopping_cart SET quantity = %s WHERE user_id = %s AND product_id = %s"
                cursor.execute(update_query, (new_quantity, user_id, product_id))
            else:
                # Insert new item
                insert_query = "INSERT INTO shopping_cart (user_id, product_id, quantity) VALUES (%s, %s, %s)"
                cursor.execute(insert_query, (user_id, product_id, quantity))
            
            conn.commit()
            cursor.close()
            return True
        except Error as e:
            logger.error(f"Error adding to cart: {e}")
            return False
        finally:
            conn.close()
    
    def get_cart(self, user_id):
        """Get shopping cart items"""
        conn = self.get_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            query = """
                SELECT sc.cart_id, p.product_id, p.name, p.price, sc.quantity, (p.price * sc.quantity) as total
                FROM shopping_cart sc
                JOIN products p ON sc.product_id = p.product_id
                WHERE sc.user_id = %s
            """
            cursor.execute(query, (user_id,))
            results = cursor.fetchall()
            cursor.close()
            return results
        except Error as e:
            logger.error(f"Error retrieving cart: {e}")
            return []
        finally:
            conn.close()
    
    def remove_from_cart(self, user_id, product_id):
        """Remove item from cart"""
        conn = self.get_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            query = "DELETE FROM shopping_cart WHERE user_id = %s AND product_id = %s"
            cursor.execute(query, (user_id, product_id))
            conn.commit()
            cursor.close()
            return True
        except Error as e:
            logger.error(f"Error removing from cart: {e}")
            return False
        finally:
            conn.close()
    
    def clear_cart(self, user_id):
        """Clear entire shopping cart"""
        conn = self.get_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            query = "DELETE FROM shopping_cart WHERE user_id = %s"
            cursor.execute(query, (user_id,))
            conn.commit()
            cursor.close()
            return True
        except Error as e:
            logger.error(f"Error clearing cart: {e}")
            return False
        finally:
            conn.close()
    
    # ============ Orders ============
    def create_order(self, user_id, cart_items):
        """Create an order from cart items"""
        conn = self.get_connection()
        if not conn:
            return None
        
        try:
            cursor = conn.cursor()
            
            # Calculate total price
            total_price = sum(item[4] * item[5] for item in cart_items)  # price * quantity
            
            # Create order
            order_query = "INSERT INTO orders (user_id, total_price, status) VALUES (%s, %s, 'pending')"
            cursor.execute(order_query, (user_id, total_price))
            order_id = cursor.lastrowid
            
            # Add order items
            for item in cart_items:
                _, product_id, name, price, quantity, _ = item
                item_query = "INSERT INTO order_items (order_id, product_id, quantity, price) VALUES (%s, %s, %s, %s)"
                cursor.execute(item_query, (order_id, product_id, quantity, price))
            
            # Clear cart
            clear_query = "DELETE FROM shopping_cart WHERE user_id = %s"
            cursor.execute(clear_query, (user_id,))
            
            conn.commit()
            cursor.close()
            return order_id
        except Error as e:
            logger.error(f"Error creating order: {e}")
            return None
        finally:
            conn.close()
    
    def get_user_orders(self, user_id):
        """Get user's order history"""
        conn = self.get_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            query = """
                SELECT order_id, total_price, status, created_at 
                FROM orders 
                WHERE user_id = %s 
                ORDER BY created_at DESC
            """
            cursor.execute(query, (user_id,))
            results = cursor.fetchall()
            cursor.close()
            return results
        except Error as e:
            logger.error(f"Error retrieving orders: {e}")
            return []
        finally:
            conn.close()
    
    def get_order_details(self, order_id):
        """Get details of a specific order"""
        conn = self.get_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            query = """
                SELECT oi.product_id, p.name, oi.quantity, oi.price, (oi.quantity * oi.price) as total
                FROM order_items oi
                JOIN products p ON oi.product_id = p.product_id
                WHERE oi.order_id = %s
            """
            cursor.execute(query, (order_id,))
            results = cursor.fetchall()
            cursor.close()
            return results
        except Error as e:
            logger.error(f"Error retrieving order details: {e}")
            return []
        finally:
            conn.close()
