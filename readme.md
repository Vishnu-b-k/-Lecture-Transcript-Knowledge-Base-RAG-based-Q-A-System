# 📚 StudyBuddy — Lecture Transcript Knowledge Base & RAG Q&A System

An AI‑powered learning assistant that transforms lecture PDFs into interactive study sessions with Q&A, quizzes, and progress tracking.

## ✨ Features

- **PDF Analysis** — Upload multiple PDFs; auto‑extract topics, stats, and summaries
- **AI Q&A** — Ask questions in natural language and get context‑aware answers
- **Smart Quizzes** — Auto‑generated multiple‑choice quizzes with explanations
- **Learning Progress** — Track which topics you've explored
- **Export** — Download Q&A history for offline review
- **Beautiful UI** — Pastel aesthetic with animations, chat bubbles, and a college logo placeholder

## 🗂️ Folder Structure

```
├── .streamlit/config.toml   # Theme & server config
├── config.py                # API keys, embedder, ChromaDB client
├── pdf_processor.py         # PDF text extraction & passage splitting
├── rag_engine.py            # Embeddings, retrieval, LLM calls, quizzes
├── transcriptor.py          # Main Streamlit UI
├── requirements.txt         # Python dependencies
├── .env.example             # Template for API key
├── .gitignore               # Files to exclude from git
└── README.md                # This file
```

## 🚀 Quick Start (Local)

```bash
# 1. Clone the repo
git clone https://github.com/Vishnu-b-k/-Lecture-Transcript-Knowledge-Base-RAG-based-Q-A-System.git
cd -Lecture-Transcript-Knowledge-Base-RAG-based-Q-A-System

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
#    Option A — environment variable (recommended):
set OPENROUTER_API_KEY=sk-or-v1-YOUR_KEY_HERE       # Windows
# export OPENROUTER_API_KEY=sk-or-v1-YOUR_KEY_HERE   # macOS / Linux

#    Option B — create a .env file:
cp .env.example .env
#    Then edit .env and paste your key

# 5. Run the app
streamlit run transcriptor.py
```

## 🌐 Deployment

### Option 1 — Streamlit Community Cloud (Free & Easiest)

1. Push your code to a **public** GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"** → Select your repo, branch, and set the main file to `transcriptor.py`.
4. Under **Advanced settings → Secrets**, add:
   ```
   OPENROUTER_API_KEY = "sk-or-v1-YOUR_KEY_HERE"
   ```
5. Click **Deploy** — your app will be live at `https://your-app.streamlit.app`.

### Option 2 — Render (Free tier available)

1. Push code to GitHub.
2. Create a **New Web Service** on [render.com](https://render.com).
3. Connect your GitHub repo.
4. Set:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run transcriptor.py --server.port $PORT --server.address 0.0.0.0`
5. Add environment variable: `OPENROUTER_API_KEY` = your key.
6. Deploy.

### Option 3 — Railway

1. Push code to GitHub.
2. Create a new project on [railway.app](https://railway.app) and connect the repo.
3. Add environment variable `OPENROUTER_API_KEY`.
4. Add a `Procfile` (if needed):
   ```
   web: streamlit run transcriptor.py --server.port $PORT --server.address 0.0.0.0
   ```
5. Deploy.

> **Note:** The `chroma_db/` folder is auto‑created at runtime. On free‑tier platforms the vector database resets on each cold start — this is fine since embeddings are rebuilt from uploaded PDFs.

## ⚙️ Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | Your OpenRouter API key | *(required)* |
| Model | Set in `config.py` → `MODEL_NAME` | `google/gemini-2.0-flash-001` |

Get your API key at [openrouter.ai/keys](https://openrouter.ai/keys).

## 📦 Dependencies

- **Streamlit** — UI framework
- **ChromaDB** — Vector database
- **Sentence Transformers** — Embedding model (`all-MiniLM-L6-v2`)
- **OpenAI (via OpenRouter)** — LLM for Q&A, quizzes, summaries
- **pdfplumber** — PDF text extraction
