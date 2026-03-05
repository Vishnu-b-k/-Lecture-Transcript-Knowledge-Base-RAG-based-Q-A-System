"""
Configuration module — API keys, model settings, and shared resources.
"""
import os

# Load .env file if present (before any other imports that might need env vars)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — rely on real env vars

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# API Configuration
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
API_BASE = "https://openrouter.ai/api/v1"
MODEL_NAME = "google/gemini-2.0-flash-001"

# ---------------------------------------------------------------------------
# Cached heavy resources — loaded once, reused across reruns
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedder():
    """Load the sentence‑transformer model exactly once."""
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def get_chroma_client():
    """Create a persistent ChromaDB client exactly once."""
    return chromadb.PersistentClient(path="./chroma_db")
