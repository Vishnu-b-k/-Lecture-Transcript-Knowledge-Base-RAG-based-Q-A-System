"""
Lecture Transcript Insights — AI‑powered study assistant for online class students.
"""
import streamlit as st
import datetime, base64, hashlib, io
import plotly.graph_objects as go
import numpy as np

from config import get_embedder, get_chroma_client
from pdf_processor import extract_text, split_passages
from rag_engine import (
    build_collection,
    get_answer,
    generate_suggestions,
    extract_topics,
    generate_summary,
    generate_quiz,
)

st.set_page_config(
    page_title="Lecture Transcript Insights",
    page_icon="assets/logo/Christ_logo.png",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
#MainMenu, footer, header {visibility: hidden;}

.logo-box {
    width: 100%; text-align: center; padding: 18px 0 10px;
}
.logo-box img {
    max-height: 80px; border-radius: 12px;
    box-shadow: 0 2px 12px rgba(91,141,239,.15);
}

.hero {
    text-align: center; padding: 24px 0 10px;
}
.hero h1 {
    background: linear-gradient(135deg, #3A7BD5, #6CB4EE);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-weight: 700; font-size: 2rem; margin: 0;
}
.hero p {
    color: #7C8DB0; font-size: .92rem; margin-top: 6px; line-height: 1.5;
}

.stat-row { display: flex; gap: 12px; margin: 12px 0; }
.stat-card {
    flex: 1; background: linear-gradient(135deg, #EDF2FA 0%, #E3EBF6 100%);
    border-radius: 14px; padding: 14px; text-align: center;
    box-shadow: 0 1px 6px rgba(91,141,239,.06);
    transition: transform .2s;
}
.stat-card:hover { transform: translateY(-2px); }
.stat-card .num {
    font-size: 1.5rem; font-weight: 700; color: #3A7BD5;
}
.stat-card .label {
    font-size: .75rem; color: #7C8DB0; margin-top: 2px;
}

.chat-q, .chat-a {
    padding: 14px 18px; border-radius: 16px; margin: 6px 0;
    font-size: .9rem; line-height: 1.6; animation: fadeUp .3s ease;
    max-width: 90%;
}
.chat-q {
    background: linear-gradient(135deg, #3A7BD5, #5B8DEF);
    color: #fff; margin-left: auto; border-bottom-right-radius: 4px;
    text-align: right;
}
.chat-a {
    background: #fff; color: #2C3E50;
    border: 1px solid #D6E4F0; border-bottom-left-radius: 4px;
    box-shadow: 0 1px 6px rgba(91,141,239,.05);
}

.tag-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0; }
.topic-tag {
    background: linear-gradient(135deg, #EDF2FA, #E3EBF6);
    color: #3A7BD5; padding: 6px 14px; border-radius: 20px;
    font-size: .78rem; font-weight: 500;
    border: 1px solid rgba(91,141,239,.12);
    transition: all .25s;
}
.topic-tag:hover {
    background: linear-gradient(135deg, #3A7BD5, #5B8DEF);
    color: #fff; transform: scale(1.05);
}

.quiz-card {
    background: #fff; border-radius: 14px; padding: 18px;
    border: 1px solid #D6E4F0; margin: 8px 0;
    box-shadow: 0 1px 8px rgba(91,141,239,.05);
}
.quiz-card h4 { color: #3A7BD5; margin: 0 0 8px; font-size: .95rem; }

.score-badge {
    display: inline-block; background: linear-gradient(135deg, #3A7BD5, #6CB4EE);
    color: #fff; font-size: 1.2rem; font-weight: 700;
    padding: 12px 28px; border-radius: 28px;
    box-shadow: 0 4px 18px rgba(58,123,213,.2);
    animation: popIn .4s ease;
}

.prog-wrap { margin: 6px 0; }
.prog-label { font-size: .75rem; color: #7C8DB0; margin-bottom: 2px; }
.prog-bar-bg {
    height: 7px; background: #E3EBF6; border-radius: 4px; overflow: hidden;
}
.prog-bar-fill {
    height: 100%; border-radius: 4px;
    background: linear-gradient(90deg, #3A7BD5, #6CB4EE);
    transition: width .6s ease;
}

.empty-state {
    text-align: center; padding: 60px 20px; color: #A0B0C8;
}
.empty-icon {
    width: 80px; height: 80px; margin: 0 auto 20px;
    background: linear-gradient(135deg, #EDF2FA, #D6E4F0);
    border-radius: 50%; display: flex; align-items: center; justify-content: center;
    font-size: 2rem; color: #5B8DEF;
}
.empty-state h3 { color: #5B8DEF; font-weight: 600; margin: 0; font-size: 1.15rem; }
.empty-state p { font-size: .88rem; margin-top: 8px; color: #7C8DB0; }

.section-title {
    color: #3A7BD5; font-weight: 600; font-size: 1.05rem;
    margin: 18px 0 8px; padding-bottom: 4px;
    border-bottom: 2px solid #EDF2FA;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes popIn {
    0%   { transform: scale(.6); opacity: 0; }
    80%  { transform: scale(1.06); }
    100% { transform: scale(1); opacity: 1; }
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F7F9FC 0%, #EDF2FA 100%) !important;
}
section[data-testid="stSidebar"] .stButton > button {
    border-radius: 10px !important;
}

.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #3A7BD5, #5B8DEF) !important;
    color: #fff !important; border: none !important;
    border-radius: 12px !important; font-weight: 600 !important;
    padding: 10px 0 !important;
    box-shadow: 0 3px 14px rgba(58,123,213,.2) !important;
    transition: transform .2s, box-shadow .2s !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 5px 20px rgba(58,123,213,.3) !important;
}

/* ── Viewport meta for mobile ── */
@viewport { width: device-width; }

/* ── Touch‑friendly defaults ── */
button, input, select, textarea {
    font-size: 16px !important; /* prevents iOS zoom on focus */
}
.stTextInput > div > div > input {
    font-size: 16px !important;
    padding: 12px !important;
}

/* ── Tablet (≤ 900px) ── */
@media (max-width: 900px) {
    .hero h1 { font-size: 1.6rem; }
    .hero p { font-size: .85rem; }
    .stat-row { gap: 8px; }
    .stat-card { padding: 10px; border-radius: 12px; }
    .stat-card .num { font-size: 1.3rem; }
    .chat-q, .chat-a { max-width: 95%; padding: 12px 14px; font-size: .88rem; }
    .section-title { font-size: .95rem; }
    .quiz-card { padding: 14px; }
    .score-badge { font-size: 1rem; padding: 10px 22px; }
}

/* ── Phone (≤ 600px) ── */
@media (max-width: 600px) {
    .hero { padding: 14px 8px 6px; }
    .hero h1 { font-size: 1.3rem; }
    .hero p { font-size: .8rem; }
    .hero p br { display: none; }
    .stat-row { flex-direction: column; gap: 6px; }
    .stat-card { padding: 10px 8px; border-radius: 10px; }
    .stat-card .num { font-size: 1.2rem; }
    .stat-card .label { font-size: .7rem; }
    .chat-q, .chat-a {
        max-width: 100%; padding: 10px 12px;
        font-size: .85rem; border-radius: 12px;
    }
    .chat-q { border-bottom-right-radius: 3px; }
    .chat-a { border-bottom-left-radius: 3px; }
    .tag-row { gap: 6px; }
    .topic-tag { padding: 5px 10px; font-size: .72rem; }
    .section-title { font-size: .9rem; margin: 12px 0 6px; }
    .quiz-card { padding: 12px; border-radius: 10px; }
    .quiz-card h4 { font-size: .88rem; }
    .score-badge { font-size: .95rem; padding: 10px 20px; border-radius: 22px; }
    .empty-state { padding: 36px 12px; }
    .empty-icon { width: 60px; height: 60px; font-size: 1.5rem; }
    .empty-state h3 { font-size: 1rem; }
    .empty-state p { font-size: .82rem; }
    .prog-label { font-size: .7rem; }
    .prog-bar-bg { height: 6px; }
}

/* ── Small phone (≤ 400px) ── */
@media (max-width: 400px) {
    .hero h1 { font-size: 1.1rem; }
    .hero p { font-size: .75rem; }
    .chat-q, .chat-a { font-size: .82rem; padding: 8px 10px; }
    .stat-card .num { font-size: 1.1rem; }
}

/* ── Streamlit columns responsive fix ── */
@media (max-width: 640px) {
    [data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
}
</style>
""", unsafe_allow_html=True)

# Inject viewport meta tag for proper mobile scaling
st.markdown(
    '<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">',
    unsafe_allow_html=True,
)

_DEFAULTS = {
    "qa_history": [],
    "collection_key": None,
    "all_passages": [],
    "pdf_text": "",
    "topics": [],
    "learning_progress": {},
    "current_quiz": [],
    "quiz_answers": {},
    "quiz_submitted": False,
    "summaries": {"brief": "", "detailed": "", "comprehensive": ""},
    "suggested_questions": [],
    "pdf_metadata": {},
    "pdf_images": [],
    "questions_per_topic": {},
    "quiz_scores": [],
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

with st.sidebar:
    st.markdown('<div class="logo-box">', unsafe_allow_html=True)
    st.image("assets/logo/Christ_logo.png", width=300)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Upload Transcripts")
    uploaded_files = st.file_uploader(
        "Drop your lecture transcript PDFs",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

st.markdown("""
<div class="hero">
    <h1>Lecture Transcript Insights</h1>
    <p>Upload your lecture transcripts from online classes and unlock key insights,<br>
    ask questions, generate quizzes, and track your learning progress.</p>
</div>
""", unsafe_allow_html=True)

collection = None

if uploaded_files:
    file_bytes_map = {}
    for f in uploaded_files:
        raw = f.read()
        f.seek(0)
        file_bytes_map[f.name] = raw

    combo_hash = hashlib.sha256(b"".join(sorted(file_bytes_map.values()))).hexdigest()[:12]

    if st.session_state.collection_key != combo_hash:
        with st.spinner("Analyzing your lecture transcripts..."):
            all_passages = []
            combined_text = ""
            pdf_meta = {}
            all_images = []

            for name, raw_bytes in file_bytes_map.items():
                text, meta, images = extract_text(raw_bytes, name)
                pdf_meta[name] = meta
                combined_text += text + "\n\n"
                all_passages.extend(split_passages(text))
                all_images.extend(images)

                # Add image captions to knowledge base so they're searchable
                for img in images:
                    cap = img.get("caption", "")
                    if cap and cap != f"Figure on page {img['page']}":
                        all_passages.append(f"[Image on page {img['page']}] {cap}")

            st.session_state.all_passages = all_passages
            st.session_state.pdf_text = combined_text
            st.session_state.pdf_metadata = pdf_meta
            st.session_state.pdf_images = all_images
            st.session_state.collection_key = combo_hash

            collection = build_collection(all_passages, combo_hash)

            st.session_state.topics = extract_topics(combined_text[:4000])
            st.session_state.learning_progress = {t: 0 for t in st.session_state.topics}
            st.session_state.suggested_questions = generate_suggestions(combined_text[:4000])
            st.session_state.summaries["brief"] = generate_summary(combined_text[:4000], "brief")
            st.session_state.summaries["detailed"] = generate_summary(combined_text[:4000], "detailed")
    else:
        collection = build_collection(st.session_state.all_passages, combo_hash)

if st.session_state.all_passages:
    with st.sidebar:
        st.markdown("---")

        total_pages = sum(m["pages"] for m in st.session_state.pdf_metadata.values())
        total_words = sum(m["word_count"] for m in st.session_state.pdf_metadata.values())
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-card">
                <div class="num">{len(st.session_state.pdf_metadata)}</div>
                <div class="label">Files</div>
            </div>
            <div class="stat-card">
                <div class="num">{total_pages}</div>
                <div class="label">Pages</div>
            </div>
            <div class="stat-card">
                <div class="num">{total_words:,}</div>
                <div class="label">Words</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.topics:
            st.markdown("### Key Topics")
            tags_html = "".join(
                f'<span class="topic-tag">{t}</span>' for t in st.session_state.topics
            )
            st.markdown(f'<div class="tag-row">{tags_html}</div>', unsafe_allow_html=True)

        if st.session_state.pdf_images:
            st.markdown(f"### Extracted Figures ({len(st.session_state.pdf_images)})")
            for idx, img in enumerate(st.session_state.pdf_images):
                with st.expander(f"Page {img['page']} — {img['caption'][:50]}", expanded=False):
                    st.image(io.BytesIO(img["bytes"]), caption=img["caption"], use_container_width=True)

        if any(v > 0 for v in st.session_state.learning_progress.values()):
            st.markdown("### Learning Progress")
            for topic, prog in st.session_state.learning_progress.items():
                if prog > 0:
                    st.markdown(f"""
                    <div class="prog-wrap">
                        <div class="prog-label">{topic} — {prog}%</div>
                        <div class="prog-bar-bg"><div class="prog-bar-fill" style="width:{prog}%"></div></div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("### Summaries")
        level = st.radio("Detail level:", ["Brief", "Detailed", "Comprehensive"],
                         horizontal=True, label_visibility="collapsed")
        level_key = level.lower()

        if level_key == "comprehensive" and not st.session_state.summaries["comprehensive"]:
            if st.button("Generate comprehensive summary"):
                st.session_state.summaries["comprehensive"] = generate_summary(
                    st.session_state.pdf_text[:4000], "comprehensive"
                )

        if st.session_state.summaries.get(level_key):
            st.markdown(st.session_state.summaries[level_key])

if not uploaded_files:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#5B8DEF" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                <polyline points="14 2 14 8 20 8"/>
                <line x1="16" y1="13" x2="8" y2="13"/>
                <line x1="16" y1="17" x2="8" y2="17"/>
                <polyline points="10 9 9 9 8 9"/>
            </svg>
        </div>
        <h3>Welcome to Lecture Transcript Insights</h3>
        <p>Upload your lecture transcript PDFs from the sidebar to get started.<br>
        Get key insights, ask questions, and test your understanding.</p>
    </div>
    """, unsafe_allow_html=True)

else:
    if st.session_state.suggested_questions:
        st.markdown('<div class="section-title">Suggested Questions</div>', unsafe_allow_html=True)
        cols = st.columns(min(len(st.session_state.suggested_questions), 3))
        for i, q in enumerate(st.session_state.suggested_questions):
            with cols[i % len(cols)]:
                if st.button(q, key=f"sq_{i}", use_container_width=True):
                    st.session_state["_pending_q"] = q

    st.markdown('<div class="section-title">Ask a Question</div>', unsafe_allow_html=True)
    with st.form("qa_form", clear_on_submit=True):
        question_input = st.text_input(
            "question",
            placeholder="e.g. What were the main topics covered in this lecture?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask", type="primary", use_container_width=True)

    pending_q = st.session_state.pop("_pending_q", None)
    active_question = pending_q or (question_input if submitted else None)

    if active_question and collection:
        if not st.session_state.qa_history or active_question != st.session_state.qa_history[-1][0]:
            with st.spinner("Finding answer from your transcripts..."):
                answer = get_answer(active_question, collection)
            st.session_state.qa_history.append((active_question, answer))

            for topic in st.session_state.topics:
                if topic.lower() in (active_question + answer).lower():
                    old = st.session_state.learning_progress.get(topic, 0)
                    st.session_state.learning_progress[topic] = min(old + 10, 100)
                    st.session_state.questions_per_topic[topic] = st.session_state.questions_per_topic.get(topic, 0) + 1

    if st.session_state.qa_history:
        st.markdown('<div class="section-title">Conversation</div>', unsafe_allow_html=True)
        for q, a in st.session_state.qa_history:
            st.markdown(f'<div class="chat-q">{q}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-a">{a}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Test Your Knowledge</div>', unsafe_allow_html=True)

    quiz_col1, quiz_col2 = st.columns([2, 1])
    with quiz_col1:
        quiz_topic = st.selectbox(
            "Pick a topic (or leave blank for general quiz):",
            options=["All topics"] + st.session_state.topics,
            label_visibility="collapsed",
        )
    with quiz_col2:
        gen_quiz = st.button("Generate Quiz", use_container_width=True)

    if gen_quiz and st.session_state.all_passages:
        topic_arg = None if quiz_topic == "All topics" else quiz_topic
        with st.spinner("Creating quiz from your transcripts..."):
            st.session_state.current_quiz = generate_quiz(st.session_state.all_passages, topic_arg)
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False

    if st.session_state.current_quiz:
        quiz_data = st.session_state.current_quiz

        if not st.session_state.quiz_submitted:
            for i, q in enumerate(quiz_data):
                st.markdown(f"""
                <div class="quiz-card">
                    <h4>Question {i+1}</h4>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"**{q['question']}**")
                options = q.get("options", ["A", "B", "C", "D"])
                selected = st.radio(
                    f"q{i+1}",
                    options=options,
                    key=f"quiz_{i}",
                    label_visibility="collapsed",
                )
                st.session_state.quiz_answers[i] = options.index(selected) if selected in options else -1

            if st.button("Submit Quiz", type="primary", use_container_width=True):
                st.session_state.quiz_submitted = True
                st.rerun()
        else:
            correct = 0
            for i, q in enumerate(quiz_data):
                user_idx = st.session_state.quiz_answers.get(i, -1)
                correct_idx = q.get("correctAnswerIndex", 0)
                is_right = user_idx == correct_idx
                if is_right:
                    correct += 1

                icon = "Correct" if is_right else "Wrong"
                color = "#27AE60" if is_right else "#E74C3C"
                st.markdown(f'<span style="color:{color};font-weight:600;">{icon}</span> — **Q{i+1}: {q["question"]}**', unsafe_allow_html=True)
                for j, opt in enumerate(q.get("options", [])):
                    if j == correct_idx:
                        st.markdown(f"&ensp; **{opt}** (correct)")
                    elif j == user_idx:
                        st.markdown(f"&ensp; ~~{opt}~~ (your answer)")
                    else:
                        st.markdown(f"&ensp;&ensp; {opt}")
                st.info(q.get("explanation", ""))

            pct = (correct / len(quiz_data)) * 100 if quiz_data else 0
            st.markdown(f"""
            <div style="text-align:center;margin:20px 0;">
                <div class="score-badge">{correct}/{len(quiz_data)} — {pct:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)

            # Track quiz score
            if f"_quiz_scored_{len(st.session_state.quiz_scores)}" not in st.session_state:
                st.session_state.quiz_scores.append({
                    "score": pct,
                    "correct": correct,
                    "total": len(quiz_data),
                    "topic": quiz_topic if quiz_topic != "All topics" else "General",
                })
                st.session_state[f"_quiz_scored_{len(st.session_state.quiz_scores)-1}"] = True

            if st.button("Try another quiz", use_container_width=True):
                st.session_state.current_quiz = []
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.rerun()

    # ── Learning Analytics with 3D Plots ──
    has_progress = any(v > 0 for v in st.session_state.learning_progress.values())
    has_quiz_data = len(st.session_state.quiz_scores) > 0

    if has_progress or has_quiz_data:
        st.markdown("---")
        st.markdown('<div class="section-title">Learning Analytics</div>', unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["3D Progress Overview", "Quiz Performance", "Topic Radar"])

        with tab1:
            topics = st.session_state.topics
            progress_vals = [st.session_state.learning_progress.get(t, 0) for t in topics]
            q_counts = [st.session_state.questions_per_topic.get(t, 0) for t in topics]

            if topics:
                fig = go.Figure(data=[go.Bar3d(
                    x=list(range(len(topics))),
                    y=[0] * len(topics),
                    z=[0] * len(topics),
                    dx=[0.6] * len(topics),
                    dy=[0.6] * len(topics),
                    dz=progress_vals,
                    color=progress_vals,
                    colorscale=[[0, '#E3EBF6'], [0.5, '#5B8DEF'], [1, '#3A7BD5']],
                    colorbar=dict(title='Progress %'),
                    hovertext=[f"{t}<br>Progress: {p}%<br>Questions: {q}" for t, p, q in zip(topics, progress_vals, q_counts)],
                    hoverinfo='text',
                )] if hasattr(go, 'Bar3d') else [go.Scatter3d(
                    x=list(range(len(topics))),
                    y=q_counts,
                    z=progress_vals,
                    mode='markers+text',
                    marker=dict(
                        size=[max(8, p/5) for p in progress_vals],
                        color=progress_vals,
                        colorscale=[[0, '#E3EBF6'], [0.5, '#5B8DEF'], [1, '#3A7BD5']],
                        colorbar=dict(title='Progress %'),
                        opacity=0.85,
                        line=dict(width=1, color='#3A7BD5'),
                    ),
                    text=topics,
                    textposition='top center',
                    hovertext=[f"{t}<br>Progress: {p}%<br>Questions asked: {q}" for t, p, q in zip(topics, progress_vals, q_counts)],
                    hoverinfo='text',
                )])

                fig.update_layout(
                    scene=dict(
                        xaxis=dict(title='Topic Index', tickvals=list(range(len(topics))), ticktext=[t[:12] for t in topics]),
                        yaxis=dict(title='Questions Asked'),
                        zaxis=dict(title='Progress %', range=[0, 110]),
                        bgcolor='#F7F9FC',
                    ),
                    margin=dict(l=0, r=0, t=30, b=0),
                    height=450,
                    paper_bgcolor='#F7F9FC',
                    title=dict(text='Topic Progress Overview', font=dict(color='#3A7BD5', size=14)),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Start asking questions to see your 3D progress chart.")

        with tab2:
            if has_quiz_data:
                scores = st.session_state.quiz_scores
                quiz_nums = list(range(1, len(scores) + 1))
                score_vals = [s['score'] for s in scores]
                score_topics = [s['topic'][:15] for s in scores]

                # 3D surface of quiz performance
                if len(scores) >= 2:
                    # Create a smooth surface from quiz history
                    x_grid = np.linspace(0, len(scores)-1, max(len(scores), 5))
                    y_grid = np.linspace(0, 100, 10)
                    X, Y = np.meshgrid(x_grid, y_grid)
                    Z = np.zeros_like(X)
                    for i, s in enumerate(scores):
                        Z += s['score'] * np.exp(-0.5 * ((X - i)**2 + (Y - s['score'])**2) / (max(len(scores), 3)))
                    Z = np.clip(Z / (Z.max() + 0.01) * 100, 0, 100)

                    fig_surface = go.Figure(data=[go.Surface(
                        x=X, y=Y, z=Z,
                        colorscale=[[0, '#E3EBF6'], [0.3, '#6CB4EE'], [0.7, '#5B8DEF'], [1, '#3A7BD5']],
                        opacity=0.88,
                        colorbar=dict(title='Intensity'),
                    )])
                    fig_surface.update_layout(
                        scene=dict(
                            xaxis=dict(title='Quiz #'),
                            yaxis=dict(title='Score Range'),
                            zaxis=dict(title='Performance', range=[0, 110]),
                            bgcolor='#F7F9FC',
                        ),
                        margin=dict(l=0, r=0, t=30, b=0),
                        height=420,
                        paper_bgcolor='#F7F9FC',
                        title=dict(text='Quiz Performance Surface', font=dict(color='#3A7BD5', size=14)),
                    )
                    st.plotly_chart(fig_surface, use_container_width=True)

                # 3D scatter of individual quizzes
                fig_scatter = go.Figure(data=[go.Scatter3d(
                    x=quiz_nums,
                    y=[s['correct'] for s in scores],
                    z=score_vals,
                    mode='markers+lines',
                    marker=dict(
                        size=10,
                        color=score_vals,
                        colorscale=[[0, '#E74C3C'], [0.5, '#F39C12'], [1, '#27AE60']],
                        colorbar=dict(title='Score %'),
                        opacity=0.9,
                        symbol='diamond',
                    ),
                    line=dict(color='#5B8DEF', width=3),
                    text=[f"Quiz {n}<br>Topic: {t}<br>Score: {s:.0f}%" for n, t, s in zip(quiz_nums, score_topics, score_vals)],
                    hoverinfo='text',
                )])
                fig_scatter.update_layout(
                    scene=dict(
                        xaxis=dict(title='Quiz Number'),
                        yaxis=dict(title='Correct Answers'),
                        zaxis=dict(title='Score %', range=[0, 110]),
                        bgcolor='#F7F9FC',
                    ),
                    margin=dict(l=0, r=0, t=30, b=0),
                    height=420,
                    paper_bgcolor='#F7F9FC',
                    title=dict(text='Quiz Scores Over Time', font=dict(color='#3A7BD5', size=14)),
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("Take a quiz to see your performance charts.")

        with tab3:
            topics = st.session_state.topics
            progress_vals = [st.session_state.learning_progress.get(t, 0) for t in topics]

            if topics and any(v > 0 for v in progress_vals):
                fig_radar = go.Figure(data=go.Scatterpolar(
                    r=progress_vals + [progress_vals[0]],
                    theta=[t[:18] for t in topics] + [topics[0][:18]],
                    fill='toself',
                    fillcolor='rgba(91,141,239,0.15)',
                    line=dict(color='#3A7BD5', width=2.5),
                    marker=dict(size=7, color='#5B8DEF'),
                    hovertext=[f"{t}<br>Progress: {p}%" for t, p in zip(topics, progress_vals)] + [""],
                    hoverinfo='text',
                ))
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor='#F7F9FC',
                        radialaxis=dict(visible=True, range=[0, 100], ticksuffix='%',
                                        gridcolor='#D6E4F0', linecolor='#D6E4F0'),
                        angularaxis=dict(gridcolor='#D6E4F0', linecolor='#D6E4F0'),
                    ),
                    margin=dict(l=40, r=40, t=40, b=40),
                    height=420,
                    paper_bgcolor='#F7F9FC',
                    title=dict(text='Topic Coverage Radar', font=dict(color='#3A7BD5', size=14)),
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.info("Start exploring topics to see your coverage radar.")

    if st.session_state.qa_history:
        st.markdown("---")
        st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)
        qa_text = "\n\n".join(f"Q: {q}\n\nA: {a}" for q, a in st.session_state.qa_history)
        fname = f"qa_history_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"
        b64 = base64.b64encode(qa_text.encode()).decode()
        st.markdown(
            f'<a href="data:file/txt;base64,{b64}" download="{fname}" '
            f'style="text-decoration:none;">'
            f'<button style="width:100%;padding:10px;border-radius:10px;border:1px solid #D6E4F0;'
            f'background:#EDF2FA;color:#3A7BD5;font-weight:600;cursor:pointer;font-family:Inter,sans-serif;">'
            f'Download Q and A History</button></a>',
            unsafe_allow_html=True,
        )