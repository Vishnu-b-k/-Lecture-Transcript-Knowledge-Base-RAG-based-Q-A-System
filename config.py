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
def _get_api_key():
    """Resolve API key: env var → Streamlit secrets → empty."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("OPENROUTER_API_KEY", "")
        except Exception:
            pass
    return key

API_KEY = _get_api_key()
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
    """Create an in-memory ChromaDB client (safe for multi-user on Streamlit Cloud)."""
    return chromadb.EphemeralClient()
