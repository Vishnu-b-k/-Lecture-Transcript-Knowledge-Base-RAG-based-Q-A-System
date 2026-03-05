"""
RAG engine — embedding, retrieval, answer generation, quizzes, summaries.
"""
import os
import re
import json
import uuid
import logging
import numpy as np
from openai import OpenAI
import streamlit as st

from config import API_KEY, API_BASE, MODEL_NAME, get_embedder, get_chroma_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_client = None

def _get_client():
    """Lazy-init the OpenAI client so env vars / secrets are ready."""
    global _client
    if _client is None:
        key = API_KEY or os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            try:
                import streamlit as _st
                key = _st.secrets.get("OPENROUTER_API_KEY", "")
            except Exception:
                pass
        _client = OpenAI(api_key=key, base_url=API_BASE)
    return _client

def _llm_call(system_prompt: str, user_prompt: str, max_tokens: int = 500, temperature: float = 0.3) -> str:
    """Single place for all LLM calls — easier to swap models later."""
    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Embedding & Indexing
# ---------------------------------------------------------------------------

def build_collection(passages: list[str], collection_key: str | None = None):
    """
    Embed passages and store in ChromaDB.
    Returns the collection object. Uses a stable key so the same set of
    passages is never re‑indexed.
    """
    embedder = get_embedder()
    client = get_chroma_client()

    if collection_key is None:
        collection_key = str(uuid.uuid4())[:8]

    collection = client.get_or_create_collection(name=f"pdfs_{collection_key}")

    # Only add if collection is empty (avoids duplicate inserts on rerun)
    if collection.count() == 0:
        embeddings = embedder.encode(passages).astype("float32")
        collection.add(
            documents=passages,
            embeddings=embeddings,
            ids=[f"{collection_key}_{i}" for i in range(len(passages))],
        )

    return collection


def query_collection(question: str, collection, n_results: int = 5) -> str:
    """Retrieve the top‑n passages most relevant to the question."""
    embedder = get_embedder()
    q_emb = embedder.encode([question]).astype("float32")
    results = collection.query(query_embeddings=q_emb, n_results=n_results, include=["documents"])
    if results["documents"]:
        return "\n".join(results["documents"][0])
    return "No relevant context found."


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def get_answer(question: str, collection) -> str:
    """Retrieve context and generate an answer."""
    context = query_collection(question, collection)
    system = (
        "You are a lecture transcript assistant. Answer questions ONLY using the provided context. "
        "If the answer is not found in the context, reply: 'This information is not available in the uploaded lecture transcripts.' "
        "Never use outside knowledge. Be concise and use markdown formatting."
    )
    user = f"Lecture transcript context:\n{context}\n\nStudent question: {question}"
    return _llm_call(system, user, max_tokens=600)


# ---------------------------------------------------------------------------
# Suggested questions
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def generate_suggestions(text_preview: str) -> list[str]:
    """Generate 5 suggested questions from the first ~4 000 chars of the PDF."""
    system = (
        "You are an expert at creating relevant questions from academic content. "
        "Generate 5 important questions that cover the key concepts. "
        "Return ONLY the questions, one per line, no numbering."
    )
    raw = _llm_call(system, text_preview[:4000], max_tokens=300, temperature=0.7)
    return [q.strip() for q in raw.split("\n") if q.strip()][:5]


# ---------------------------------------------------------------------------
# Topic extraction
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Extracting topics…")
def extract_topics(text_preview: str) -> list[str]:
    """Extract 5‑10 key topics from the text."""
    system = (
        "Extract the 5‑10 most important topics from this academic text. "
        "Return a valid JSON array of short topic strings."
    )
    raw = _llm_call(system, text_preview[:4000], max_tokens=300, temperature=0.3)
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    try:
        return json.loads(match.group(0)) if match else json.loads(raw)
    except Exception:
        return [t.strip().strip('"') for t in re.split(r",|\n", raw) if t.strip() and len(t.strip()) > 3]


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

SUMMARY_PROMPTS = {
    "brief": "Create a very brief overview (2‑3 sentences) capturing the main theme.",
    "detailed": "Create a detailed outline covering the main sections and key points (about 1 paragraph).",
    "comprehensive": "Create a comprehensive summary with all important concepts, theories, and examples (several paragraphs).",
}

@st.cache_data(show_spinner="Generating summary…")
def generate_summary(text_preview: str, level: str = "detailed") -> str:
    tokens = {"brief": 300, "detailed": 600, "comprehensive": 1200}
    return _llm_call(
        f"You are an expert summariser. {SUMMARY_PROMPTS[level]}",
        f"Text to summarise:\n\n{text_preview[:4000]}",
        max_tokens=tokens.get(level, 600),
    )


# ---------------------------------------------------------------------------
# Quiz generation
# ---------------------------------------------------------------------------

def generate_quiz(passages: list[str], topic: str | None = None) -> list[dict]:
    """Generate 5 MCQ questions as a list of dicts."""
    sample_size = min(10, len(passages))
    indices = np.random.choice(len(passages), sample_size, replace=False)
    context = "\n".join([passages[i] for i in indices])

    system = (
        f"Create 5 multiple‑choice questions about '{topic or 'the document'}'. "
        "Return a JSON array where each object has: "
        '"question", "options" (array of 4), "correctAnswerIndex" (0‑3), "explanation".'
    )
    raw = _llm_call(system, f"Content:\n{context[:4000]}", max_tokens=1500, temperature=0.7)

    match = re.search(r"\[.*\]", raw, re.DOTALL)
    try:
        return json.loads(match.group(0)) if match else json.loads(raw)
    except Exception:
        logger.warning("Quiz JSON parse failed — returning empty quiz.")
        return []
