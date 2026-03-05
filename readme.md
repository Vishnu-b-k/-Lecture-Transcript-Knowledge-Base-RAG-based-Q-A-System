# Lecture Transcript Knowledge Base — RAG-based Q&A System

An AI-powered learning assistant that transforms lecture transcript PDFs into interactive study sessions with Q&A, quizzes, summaries, and learning analytics.

**Live Demo:** [4niawbvvzyh7m2adukfyko.streamlit.app](https://4niawbvvzyh7m2adukfyko.streamlit.app/)

---

## Features

- **PDF Analysis** — Upload multiple lecture transcript PDFs; automatically extract topics, word counts, page stats, and embedded figures.
- **AI-Powered Q&A** — Ask questions in natural language and receive context-aware answers grounded in the uploaded transcripts.
- **Smart Quizzes** — Generate multiple-choice quizzes on specific topics or across all content, with scoring and explanations.
- **Multi-Level Summaries** — Brief, detailed, and comprehensive summaries generated on demand.
- **Learning Analytics** — Interactive 3D visualizations (progress overview, quiz performance surface, topic radar) built with Plotly.
- **Progress Tracking** — Monitor which topics you have explored and how many questions you have asked per topic.
- **Export** — Download your full Q&A history as a text file for offline review.

---

## Architecture

The system follows a Retrieval-Augmented Generation (RAG) pipeline:

1. **Ingestion** — PDFs are parsed with pdfplumber and PyMuPDF; text is split into overlapping passages.
2. **Embedding** — Passages are encoded using Sentence Transformers (`all-MiniLM-L6-v2`) and stored in ChromaDB.
3. **Retrieval** — User questions are embedded and matched against the vector store to find relevant context.
4. **Generation** — Retrieved context is passed to an LLM (Google Gemini 2.0 Flash via OpenRouter) to produce answers, quizzes, and summaries.

---

## Project Structure

```
project_main/
├── .streamlit/
│   └── config.toml          # Streamlit theme and server configuration
├── assets/
│   └── logo/
│       └── Christ_logo.png  # Application logo
├── config.py                # API keys, model settings, cached resources
├── pdf_processor.py         # PDF text extraction, figure extraction, passage splitting
├── rag_engine.py            # Embedding, retrieval, LLM calls, quiz generation
├── transcriptor.py          # Main Streamlit application (UI and orchestration)
├── requirements.txt         # Python dependencies
├── .env.example             # Template for environment variables
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

---

## Getting Started

### Prerequisites

- Python 3.10 or later
- An OpenRouter API key ([get one here](https://openrouter.ai/keys))

### Local Setup

```bash
# Clone the repository
git clone https://github.com/Vishnu-b-k/-Lecture-Transcript-Knowledge-Base-RAG-based-Q-A-System.git
cd -Lecture-Transcript-Knowledge-Base-RAG-based-Q-A-System

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Configure your API key (choose one method)

# Method A — Environment variable:
set OPENROUTER_API_KEY=sk-or-v1-YOUR_KEY_HERE         # Windows
# export OPENROUTER_API_KEY=sk-or-v1-YOUR_KEY_HERE     # macOS / Linux

# Method B — .env file:
cp .env.example .env
# Edit .env and add your key

# Run the application
streamlit run transcriptor.py
```

The app will open at `http://localhost:8501`.

---

## Deployment

### Streamlit Community Cloud (Recommended)

1. Push the repository to GitHub.
2. Sign in at [share.streamlit.io](https://share.streamlit.io).
3. Click **New app**, select the repository and branch, and set `transcriptor.py` as the main file.
4. Under **Advanced settings > Secrets**, add:
   ```toml
   OPENROUTER_API_KEY = "sk-or-v1-YOUR_KEY_HERE"
   ```
5. Click **Deploy**. The app will be available at `https://<your-app>.streamlit.app`.

### Render

1. Create a **New Web Service** on [render.com](https://render.com) and connect the GitHub repository.
2. Set the build command to `pip install -r requirements.txt`.
3. Set the start command to:
   ```
   streamlit run transcriptor.py --server.port $PORT --server.address 0.0.0.0
   ```
4. Add the `OPENROUTER_API_KEY` environment variable.
5. Deploy.

> **Note:** The `chroma_db/` directory is created at runtime. On free-tier platforms the vector database resets on each cold start; this is expected since embeddings are rebuilt from the uploaded PDFs.

---

## Configuration

| Variable             | Description                       | Default                         |
|----------------------|-----------------------------------|---------------------------------|
| `OPENROUTER_API_KEY` | Your OpenRouter API key           | *(required)*                    |
| `MODEL_NAME`         | LLM model (set in `config.py`)   | `google/gemini-2.0-flash-001`  |
| Embedding model      | Sentence Transformer model        | `all-MiniLM-L6-v2`             |

---

## Tech Stack

| Component         | Technology                                 |
|-------------------|--------------------------------------------|
| Frontend          | Streamlit                                  |
| Vector Database   | ChromaDB                                   |
| Embeddings        | Sentence Transformers (all-MiniLM-L6-v2)   |
| LLM               | Google Gemini 2.0 Flash (via OpenRouter)   |
| PDF Parsing       | pdfplumber, PyMuPDF                        |
| Visualizations    | Plotly (3D scatter, surface, radar)         |

---

## License

This project was developed as part of a academic project at Christ University.
