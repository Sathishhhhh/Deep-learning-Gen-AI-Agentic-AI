"""
RAG.py - Real Estate RAG Assistant Entry Point
This is the main file to run the Streamlit application.

Run with: streamlit run RAG.py
Or use: python utils.py test  (to test the system)
"""

import subprocess
import sys
import os

if __name__ == "__main__":
    # Run the Streamlit app
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], cwd=os.path.dirname(__file__))
