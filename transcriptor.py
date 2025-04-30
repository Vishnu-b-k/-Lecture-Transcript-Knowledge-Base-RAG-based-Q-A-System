import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import tempfile
import pdfplumber
import os
import uuid
import openai
import datetime
import base64
import pyttsx3
import threading
import time
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import speech_recognition as sr
from PIL import Image
import io
import fitz  # PyMuPDF
import logging
import traceback
from collections import Counter
from wordcloud import WordCloud

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Breakpoint function for debugging
def debug_breakpoint(message, data=None):
    """Custom breakpoint that logs debugging information"""
    logger.info(f"BREAKPOINT: {message}")
    if data:
        logger.info(f"DATA: {data}")
    # You can add additional debugging functionality here if needed

# API Config
openai.api_key = "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"
openai.api_base = "https://openrouter.ai/api/v1"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")

st.set_page_config(page_title="RAG PDF Learning Assistant", layout="wide")

# CSS to improve UI
st.markdown("""
<style>
    .main-header {color:#29465B; text-align:center; font-size:2.5rem; margin-bottom:1rem;}
    .subheader {color:#3d85c6; font-size:1.5rem; margin-top:1rem;}
    .citation {background-color:#f5f5f5; padding:10px; border-left:3px solid #29465B; margin:10px 0;}
    .topic-button {background-color:#e6f2ff; padding:5px; border-radius:5px; margin:2px; cursor:pointer;}
    .progress-bar {height:20px; background-color:#e6f2ff; border-radius:5px; margin:5px 0;}
    .progress-fill {height:100%; background-color:#3d85c6; border-radius:5px;}
    .chart-container {background-color:#fff; padding:10px; border:1px solid #ddd; border-radius:5px;}
    .quiz-question {background-color:#f9f9f9; padding:15px; border-radius:5px; margin:10px 0;}
    .suggested-question {transition: all 0.3s;}
    .suggested-question:hover {background-color:#f0f9ff; transform: scale(1.02);}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>Advanced PDF Learning Assistant</h1>", unsafe_allow_html=True)

# State Initialization
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

if "tts_active" not in st.session_state:
    st.session_state.tts_active = False

if "last_question" not in st.session_state:
    st.session_state.last_question = ""

if "cache_key" not in st.session_state:
    st.session_state.cache_key = str(uuid.uuid4())

if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

if "process_question" not in st.session_state:
    st.session_state.process_question = False
    
if "question_to_process" not in st.session_state:
    st.session_state.question_to_process = ""

# New state variables for advanced features
if "extracted_topics" not in st.session_state:
    st.session_state.extracted_topics = {}  # {topic: [passage_ids]}

if "topic_counts" not in st.session_state:
    st.session_state.topic_counts = {}  # {topic: count}

if "learning_progress" not in st.session_state:
    st.session_state.learning_progress = {}  # {topic: coverage_percentage}

if "figures_tables" not in st.session_state:
    st.session_state.figures_tables = {}  # {id: {"type": "figure/table", "content": binary_data, "caption": text}}

if "current_quiz" not in st.session_state:
    st.session_state.current_quiz = []  # [{question, options, answer, explanation}]

if "pdf_metadata" not in st.session_state:
    st.session_state.pdf_metadata = {}  # {filename: {page_count, word_count, etc}}

if "summaries" not in st.session_state:
    st.session_state.summaries = {"brief": "", "detailed": "", "comprehensive": ""}

if "citation_mapping" not in st.session_state:
    st.session_state.citation_mapping = {}  # {passage_id: {page, paragraph, text}}

# Custom TTS function that runs in a separate process to avoid Streamlit rerendering
def speak_text_safe(text):
    debug_breakpoint("Starting TTS process")
    def _speak():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            logger.error(f"TTS Error: {e}")
    
    # Run in a separate thread
    t = threading.Thread(target=_speak)
    t.daemon = True
    t.start()
    return t

def listen_for_question():
    """Function to capture speech input and convert to text"""
    debug_breakpoint("Starting speech recognition")
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak your question.")
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=15)
            st.info("Processing your speech...")
            text = r.recognize_google(audio)
            return text
        except sr.WaitTimeoutError:
            st.warning("No speech detected. Please try again.")
        except sr.UnknownValueError:
            st.warning("Could not understand audio. Please try again.")
        except sr.RequestError as e:
            st.error(f"Speech recognition service error: {e}")
        except Exception as e:
            st.error(f"Error during speech recognition: {e}")
    return None

def add_answer(question, answer, citations=None):
    """Add Q&A to history with citations if available"""
    debug_breakpoint(f"Adding answer for question: {question[:30]}...")
    # Only add to history if it's a new question (prevents duplicates)
    if not st.session_state.qa_history or question != st.session_state.qa_history[-1][0]:
        st.session_state.qa_history.append((question, answer, citations))
        st.session_state.last_question = question
        
        # Update learning progress based on topics in the question and answer
        update_learning_progress(question, answer)

def update_learning_progress(question, answer):
    """Update learning progress based on topics covered in Q&A"""
    debug_breakpoint("Updating learning progress")
    combined_text = f"{question} {answer}"
    
    # Check which topics are covered in this Q&A
    for topic in st.session_state.extracted_topics:
        if topic.lower() in combined_text.lower():
            current = st.session_state.learning_progress.get(topic, 0)
            # Increment progress but cap at 100%
            st.session_state.learning_progress[topic] = min(current + 10, 100)

def read_answer(idx):
    """Read answer aloud using TTS"""
    debug_breakpoint(f"Reading answer at index {idx}")
    # Get the answer from history
    _, answer, _ = st.session_state.qa_history[idx]
    # Start TTS in a separate thread
    thread = speak_text_safe(answer)
    # Small delay to prevent immediate rerun
    time.sleep(0.1)

def generate_question_suggestions(text):
    """Generate question suggestions based on PDF content"""
    debug_breakpoint("Generating question suggestions")
    try:
        # Use OpenAI to generate questions
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at creating relevant questions from academic or lecture content. Generate 5 important questions that would help someone understand the key concepts in the provided text. Return ONLY the questions with no additional text or numbering, with one question per line."},
                {"role": "user", "content": f"Here is the text from a lecture or academic content:\n\n{text[:4000]}...\n\nGenerate 5 clear, concise questions that cover the main concepts."}
            ],
            max_tokens=300,
            temperature=0.7
        )
        questions_text = response["choices"][0]["message"]["content"].strip()
        # Split by newlines and clean up
        questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
        return questions[:5]  # Ensure we return at most 5 questions
    except Exception as e:
        logger.error(f"Error generating questions: {e}")
        return ["What is the main topic of this document?", 
                "Could you summarize the key points?",
                "What are the most important concepts covered?"]

def use_suggested_question(question):
    """Set the question to be processed and flag it for immediate processing"""
    debug_breakpoint(f"Using suggested question: {question[:30]}...")
    st.session_state.question_input = question
    st.session_state.process_question = True
    st.session_state.question_to_process = question

def extract_topics(text):
    """Extract main topics from text using OpenAI"""
    debug_breakpoint("Extracting topics from text")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at identifying main topics and concepts in academic or educational content. Extract the 5-10 most important topics from the provided text. Return the topics as a valid JSON array of strings."},
                {"role": "user", "content": f"Here is the text from academic content:\n\n{text[:4000]}...\n\nExtract the 5-10 most important topics as a JSON array."}
            ],
            max_tokens=300,
            temperature=0.3
        )
        topics_text = response["choices"][0]["message"]["content"].strip()
        
        # Try to extract JSON array
        matches = re.search(r'\[.*?\]', topics_text, re.DOTALL)
        if matches:
            topics_json = matches.group(0)
        else:
            topics_json = topics_text
            
        try:
            topics = json.loads(topics_json)
            return topics
        except:
            # Fallback if the JSON parsing fails
            # Extract anything that looks like a topic (words or phrases in quotes)
            topic_matches = re.findall(r'"([^"]+)"', topics_text)
            if topic_matches:
                return topic_matches
            else:
                # Last resort - just split by commas or newlines
                raw_topics = re.split(r',|\n', topics_text)
                return [t.strip() for t in raw_topics if t.strip() and len(t.strip()) > 3]
                
    except Exception as e:
        logger.error(f"Error extracting topics: {e}")
        # Do basic keyword extraction as fallback
        words = text.split()
        # Remove common words
        stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'is', 'are', 'was', 'were'}
        filtered = [w.lower() for w in words if w.lower() not in stopwords and len(w) > 3]
        # Count word frequencies
        word_counts = Counter(filtered)
        # Return most common words as topics
        return [word for word, _ in word_counts.most_common(10)]

def map_topics_to_passages(topics, passages):
    """Map topics to relevant passages"""
    debug_breakpoint("Mapping topics to passages")
    topic_mapping = {topic: [] for topic in topics}
    
    for i, passage in enumerate(passages):
        passage_lower = passage.lower()
        for topic in topics:
            if topic.lower() in passage_lower:
                topic_mapping[topic].append(i)
    
    return topic_mapping

def generate_summary(text, level="detailed"):
    """Generate a summary of the text at different detail levels"""
    debug_breakpoint(f"Generating {level} summary")
    
    detail_prompts = {
        "brief": "Create a very brief overview (2-3 sentences) that captures the main theme",
        "detailed": "Create a detailed outline covering the main sections and key points (about 1 paragraph)",
        "comprehensive": "Create a comprehensive summary with all important concepts, theories, and examples (several paragraphs)"
    }
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are an expert at creating concise yet informative summaries. {detail_prompts[level]} of the provided text."},
                {"role": "user", "content": f"Here is the text to summarize:\n\n{text[:4000]}..."}
            ],
            max_tokens=500 if level == "brief" else (1000 if level == "detailed" else 2000),
            temperature=0.3
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return f"Unable to generate {level} summary due to an error."

def extract_figures_and_tables(pdf_path):
    """Extract figures, tables and their captions from PDF"""
    debug_breakpoint(f"Extracting figures and tables from {pdf_path}")
    results = {}
    
    try:
        # Open the PDF with PyMuPDF
        doc = fitz.open(pdf_path)
        
        # Extract images
        image_count = 0
        for page_num, page in enumerate(doc):
            # Get images
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                # Extract image
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Get nearby text that might be a caption (simplified approach)
                text_blocks = page.get_text("blocks")
                image_rect = page.get_image_bbox(img)
                
                # Find text below the image that might be a caption
                caption = ""
                for block in text_blocks:
                    block_rect = fitz.Rect(block[:4])
                    if block_rect.y0 > image_rect.y1 and block_rect.y0 - image_rect.y1 < 50:
                        caption = block[4]
                        break
                
                if not caption:
                    caption = f"Figure on page {page_num + 1}"
                
                # Store image and caption
                image_id = f"img_{page_num}_{img_index}"
                results[image_id] = {
                    "type": "figure",
                    "content": image_bytes,
                    "caption": caption,
                    "page": page_num + 1
                }
                image_count += 1
                
                # Limit to prevent performance issues
                if image_count >= 10:
                    break
        
        # Identify table-like structures (simplified)
        for page_num, page in enumerate(doc):
            # Extract text
            text = page.get_text()
            
            # Very simple table detection - look for patterns of aligned text
            # This is a simplification - real table detection is more complex
            lines = text.split('\n')
            potential_table_start = -1
            
            for i, line in enumerate(lines):
                # Check for table indicators - multiple spaces or tabs indicating columns
                if len(line) > 10 and ('  ' in line or '\t' in line):
                    if potential_table_start == -1:
                        potential_table_start = i
                else:
                    if potential_table_start != -1 and i - potential_table_start > 2:
                        # We found what might be a table
                        table_text = '\n'.join(lines[potential_table_start:i])
                        
                        # Look for caption
                        caption = ""
                        if potential_table_start > 0 and lines[potential_table_start-1].lower().startswith("table"):
                            caption = lines[potential_table_start-1]
                        else:
                            caption = f"Table on page {page_num + 1}"
                        
                        # Store table
                        table_id = f"table_{page_num}_{potential_table_start}"
                        results[table_id] = {
                            "type": "table",
                            "content": table_text,
                            "caption": caption,
                            "page": page_num + 1
                        }
                    
                    potential_table_start = -1
        
    except Exception as e:
        logger.error(f"Error extracting figures/tables: {e}")
        traceback.print_exc()
    
    return results

def create_citation_mapping(pdf_paths, passages):
    """Create mapping from passages to their location in the source PDFs"""
    debug_breakpoint("Creating citation mapping")
    citation_map = {}
    passage_id = 0
    
    for pdf_path in pdf_paths:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if not text:
                        continue
                    
                    # Split into paragraphs
                    paragraphs = re.split(r'\n\s*\n', text)
                    
                    for para_num, paragraph in enumerate(paragraphs):
                        # Check which passages are from this paragraph
                        for i in range(passage_id, min(passage_id + len(paragraphs), len(passages))):
                            if i < len(passages) and passages[i] in paragraph:
                                citation_map[i] = {
                                    "file": os.path.basename(pdf_path),
                                    "page": page_num + 1,
                                    "paragraph": para_num + 1,
                                    "text": paragraph[:100] + "..."  # Preview
                                }
                        
                        passage_id += 1
                        
        except Exception as e:
            logger.error(f"Error creating citation mapping for {pdf_path}: {e}")
    
    return citation_map

def get_citations_for_answer(question, collection):
    """Get relevant citations for an answer"""
    debug_breakpoint("Getting citations for answer")
    try:
        query_embedding = embedder.encode([question]).astype("float32")
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=3,
            include=["documents", "metadata", "distances"]
        )
        
        citations = []
        for i, doc_id in enumerate(results.get("ids", [[]])[0]):
            # Extract the numeric ID from the doc_id string (e.g., "abc123_42" -> 42)
            try:
                passage_id = int(doc_id.split("_")[-1])
                citation_info = st.session_state.citation_mapping.get(passage_id, {})
                
                if citation_info:
                    citations.append({
                        "id": passage_id,
                        "file": citation_info.get("file", "Unknown"),
                        "page": citation_info.get("page", "Unknown"),
                        "relevance": 1 - results["distances"][0][i],  # Convert distance to relevance
                        "preview": citation_info.get("text", "")
                    })
            except:
                continue
                
        return citations
    except Exception as e:
        logger.error(f"Error getting citations: {e}")
        return []

def process_query(question, collection):
    """Process the query and get an answer with citations"""
    debug_breakpoint(f"Processing query: {question[:30]}...")
    
    with st.spinner("Generating answer..."):
        # Get relevant passages
        query_embedding = embedder.encode([question]).astype("float32")
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=5,
            include=["documents"]
        )
        context = "\n".join(results["documents"][0]) if results["documents"] else "No relevant context found."

        # Get citations
        citations = get_citations_for_answer(question, collection)

        try:
            # Check if the question asks for a visual representation
            requires_visualization = any(term in question.lower() for term in 
                                     ["graph", "chart", "plot", "visualize", "diagram", "compare", "trend", "statistics"])

            # Include visualization instructions if needed
            viz_instruction = ""
            if requires_visualization:
                viz_instruction = """
                If the question asks for numerical comparisons or statistical information, include a section at the end with 
                structured data for visualization in this format:
                [VISUALIZATION_DATA]
                {"type": "bar|line|pie", "title": "Chart Title", "data": {"labels": ["A", "B", "C"], "values": [1, 2, 3]}}
                [/VISUALIZATION_DATA]
                """

            # Get answer from OpenAI
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are a helpful academic assistant answering questions based on lecture content. Be concise but thorough. {viz_instruction}"},
                    {"role": "user", "content": f"Context from the document:\n{context}\n\nQuestion: {question}"}
                ],
                max_tokens=500,
                temperature=0.3
            )
            answer = response["choices"][0]["message"]["content"]
            
            # Add answer to history with citations
            add_answer(question, answer, citations)
            return True
        except Exception as e:
            st.error(f"LLM Error: {e}")
            logger.error(f"LLM Error: {e}")
            return False

def generate_quiz(collection, topic=None):
    """Generate a quiz based on the document content, optionally focusing on a specific topic"""
    debug_breakpoint(f"Generating quiz for topic: {topic if topic else 'all'}")
    
    try:
        # Get relevant passages for the topic if specified
        context = ""
        if topic and topic in st.session_state.extracted_topics:
            passage_ids = st.session_state.extracted_topics[topic]
            # Get these passages from the collection
            passages = [st.session_state.all_passages[pid] for pid in passage_ids if pid < len(st.session_state.all_passages)]
            context = "\n".join(passages)
        else:
            # Use a sample of passages from the entire document
            sample_size = min(10, len(st.session_state.all_passages))
            passage_indices = np.random.choice(len(st.session_state.all_passages), sample_size, replace=False)
            context = "\n".join([st.session_state.all_passages[i] for i in passage_indices])
        
        # Generate quiz questions using OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at creating educational assessment questions. Create 5 multiple-choice questions based on the provided content. For each question, provide 4 options with only one correct answer, and include a brief explanation for why the answer is correct. Return the quiz in JSON format as an array of objects with 'question', 'options' (array), 'correctAnswerIndex', and 'explanation' fields."},
                {"role": "user", "content": f"Here is the content to base the quiz on:\n\n{context[:4000]}...\n\nCreate 5 multiple-choice questions on this material in JSON format."}
            ],
            max_tokens=1500,
            temperature=0.7
        )
        
        quiz_text = response["choices"][0]["message"]["content"]
        
        # Try to extract JSON
        json_match = re.search(r'\[.*\]', quiz_text, re.DOTALL)
        if json_match:
            quiz_json = json_match.group(0)
        else:
            quiz_json = quiz_text
            
        try:
            quiz_data = json.loads(quiz_json)
            return quiz_data
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            st.warning("Could not parse quiz properly. Generating a simpler version.")
            
            # Create a simpler quiz with basic questions
            simple_quiz = []
            
            # Extract questions and options with regex
            q_matches = re.findall(r'(?:Question|Q):?\s*(.*?)(?:\?|$)', quiz_text)
            options_matches = re.findall(r'(?:Options|A|B|C|D):?\s*(.*?)(?:\n|$)', quiz_text)
            answer_matches = re.findall(r'(?:Answer|Correct):?\s*(.*?)(?:\n|$)', quiz_text)
            
            for i, q in enumerate(q_matches[:5]):
                if i < len(q_matches):
                    question = q_matches[i] + "?"
                    options = ["Option A", "Option B", "Option C", "Option D"]
                    
                    # Try to get real options if available
                    opt_start = 4 * i
                    if opt_start + 4 <= len(options_matches):
                        options = options_matches[opt_start:opt_start+4]
                    
                    # Default to first option as correct if can't determine
                    correct_idx = 0
                    if i < len(answer_matches):
                        # Try to extract answer index from text like "A" or "Option A"
                        ans_text = answer_matches[i].upper()
                        if "A" in ans_text:
                            correct_idx = 0
                        elif "B" in ans_text:
                            correct_idx = 1
                        elif "C" in ans_text:
                            correct_idx = 2
                        elif "D" in ans_text:
                            correct_idx = 3
                    
                    simple_quiz.append({
                        "question": question,
                        "options": options,
                        "correctAnswerIndex": correct_idx,
                        "explanation": "This is the correct answer based on the content."
                    })
            
            return simple_quiz
            
    except Exception as e:
        logger.error(f"Error generating quiz: {e}")
        return []

def create_visualization(data_str):
    """Create visualization based on structured data in the answer"""
    debug_breakpoint("Creating visualization")
    
    try:
        # Find visualization data in the answer
        viz_match = re.search(r'\[VISUALIZATION_DATA\](.*?)\[/VISUALIZATION_DATA\]', data_str, re.DOTALL)
        if not viz_match:
            return None
            
        viz_data_str = viz_match.group(1).strip()
        
        # Parse the JSON data
        viz_data = json.loads(viz_data_str)
        
        # Create a matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        chart_type = viz_data.get('type', 'bar')
        title = viz_data.get('title', 'Data Visualization')
        labels = viz_data['data']['labels']
        values = viz_data['data']['values']
        
        if chart_type == 'bar':
            ax.bar(labels, values)
        elif chart_type == 'line':
            ax.plot(labels, values, marker='o')
        elif chart_type == 'pie':
            ax.pie(values, labels=labels, autopct='%1.1f%%')
        
        ax.set_title(title)
        
        if chart_type != 'pie':
            ax.set_xlabel('Categories')
            ax.set_ylabel('Values')
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        return None

# Main application logic
st.sidebar.title("PDF Learning Assistant")

# Set up the main application layout with sidebar
uploaded_files = st.sidebar.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)

debug_breakpoint(f"Number of uploaded files: {len(uploaded_files) if uploaded_files else 0}")

# Store all passages and their PDF paths
st.session_state.all_passages = []
pdf_paths = []
combined_text = ""

if uploaded_files:
    with st.spinner("Processing uploaded PDFs..."):
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                pdf_path = tmp.name
                pdf_paths.append(pdf_path)

            debug_breakpoint(f"Processing PDF: {file.name}")
            
            # Extract figures and tables
            figures_tables = extract_figures_and_tables(pdf_path)
            st.session_state.figures_tables.update(figures_tables)
            
            # Extract text
            with pdfplumber.open(pdf_path) as pdf:
                page_texts = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        page_texts.append(text)
                
                pdf_text = "\n".join(page_texts)
                combined_text += pdf_text + "\n\n"
                
                # Store metadata
                st.session_state.pdf_metadata[file.name] = {
                    "pages": len(pdf.pages),
                    "word_count": len(pdf_text.split()),
                    "char_count": len(pdf_text)
                }

            def split_passages(text, max_length=500):
                """Split text into passages of approximately equal length"""
                words = text.split()
                passages, chunk = [], []
                for word in words:
                    if len(" ".join(chunk)) + len(word) + 1 <= max_length:
                        chunk.append(word)
                    else:
                        passages.append(" ".join(chunk))
                        chunk = [word]
                if chunk:
                    passages.append(" ".join(chunk))
                return passages

            # Add passages from this PDF
            pdf_passages = split_passages(pdf_text)
            st.session_state.all_passages.extend(pdf_passages)

        debug_breakpoint(f"Total passages extracted: {len(st.session_state.all_passages)}")
        
       # Generate question suggestions if we have new content
        if combined_text and combined_text != st.session_state.pdf_text:
            st.session_state.pdf_text = combined_text
            
            with st.spinner("Analyzing content..."):
                # Extract topics
                topics = extract_topics(combined_text)
                debug_breakpoint(f"Extracted topics: {topics}")
                
                # Map topics to passages
                topic_mapping = map_topics_to_passages(topics, st.session_state.all_passages)
                st.session_state.extracted_topics = topic_mapping
                
                # Initialize learning progress for each topic
                st.session_state.learning_progress = {topic: 0 for topic in topics}
                
                # Create citation mapping
                st.session_state.citation_mapping = create_citation_mapping(pdf_paths, st.session_state.all_passages)
                
                # Generate summaries at different levels
                st.session_state.summaries["brief"] = generate_summary(combined_text, "brief")
                st.session_state.summaries["detailed"] = generate_summary(combined_text, "detailed")
                # Comprehensive summary can be generated on demand to save time
                
                # Generate suggested questions
                st.session_state.suggested_questions = generate_question_suggestions(combined_text)

        # Create embeddings and store in ChromaDB
        debug_breakpoint("Creating embeddings")
        embeddings = embedder.encode(st.session_state.all_passages).astype("float32")
        db_id = str(uuid.uuid4())[:8]
        collection = client.get_or_create_collection(name=f"pdfs_{db_id}")
        collection.add(
            documents=st.session_state.all_passages,
            embeddings=embeddings,
            ids=[f"{db_id}_{i}" for i in range(len(st.session_state.all_passages))],
        )
        st.success(f"{len(uploaded_files)} PDF(s) indexed successfully with {len(st.session_state.all_passages)} passages.")

    # Sidebar with topics and navigation
    if st.session_state.extracted_topics:
        st.sidebar.markdown("### Document Topics")
        st.sidebar.write("Click a topic to explore related content:")
        
        # Show topics with progress bars
        for topic, passage_ids in st.session_state.extracted_topics.items():
            progress = st.session_state.learning_progress.get(topic, 0)
            
            # Create a collapsible section for each topic
            with st.sidebar.expander(f"{topic} ({len(passage_ids)} sections)"):
                # Show progress bar
                st.markdown(f"""
                <div class="progress-bar">
                    <div class="progress-fill" style="width:{progress}%;"></div>
                </div>
                <p style="font-size:0.8rem;">Learning progress: {progress}%</p>
                """, unsafe_allow_html=True)
                
                # Button to create topic-specific questions
                if st.button(f"Explore '{topic}'", key=f"explore_{topic}"):
                    # Use OpenAI to generate a topic-specific question
                    topic_passages = [st.session_state.all_passages[pid] for pid in passage_ids[:3] 
                                     if pid < len(st.session_state.all_passages)]
                    topic_text = "\n".join(topic_passages)
                    
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": f"Create a fundamental question about '{topic}' based on this text."},
                            {"role": "user", "content": topic_text}
                        ],
                        max_tokens=50,
                        temperature=0.7
                    )
                    topic_question = response["choices"][0]["message"]["content"].strip()
                    use_suggested_question(topic_question)
                
                # Button to generate topic quiz
                if st.button(f"Quiz on '{topic}'", key=f"quiz_{topic}"):
                    st.session_state.current_quiz = generate_quiz(collection, topic)
                    st.rerun()
        
        # PDF Summary Section
        st.sidebar.markdown("### Document Summaries")
        summary_type = st.sidebar.radio("Select summary level:", ["Brief", "Detailed", "Comprehensive"])
        
        if summary_type.lower() == "comprehensive" and not st.session_state.summaries["comprehensive"]:
            if st.sidebar.button("Generate Comprehensive Summary"):
                with st.spinner("Generating comprehensive summary..."):
                    st.session_state.summaries["comprehensive"] = generate_summary(combined_text, "comprehensive")
                st.rerun()
        
        if st.session_state.summaries[summary_type.lower()]:
            st.sidebar.markdown(f"**{summary_type} Summary:**")
            st.sidebar.markdown(st.session_state.summaries[summary_type.lower()])
        
        # Figures and Tables Section
        if st.session_state.figures_tables:
            st.sidebar.markdown("### Figures & Tables")
            viz_type = st.sidebar.radio("View:", ["Figures", "Tables", "All"])
            
            items_to_show = []
            if viz_type == "Figures":
                items_to_show = [(id, item) for id, item in st.session_state.figures_tables.items() if item["type"] == "figure"]
            elif viz_type == "Tables":
                items_to_show = [(id, item) for id, item in st.session_state.figures_tables.items() if item["type"] == "table"]
            else:
                items_to_show = list(st.session_state.figures_tables.items())
            
            # Display the first few items with pagination
            items_per_page = 2
            total_pages = (len(items_to_show) + items_per_page - 1) // items_per_page
            
            if "viz_page" not in st.session_state:
                st.session_state.viz_page = 0
            
            # Page navigation
            if total_pages > 1:
                cols = st.sidebar.columns(2)
                if cols[0].button("← Previous", disabled=st.session_state.viz_page <= 0):
                    st.session_state.viz_page -= 1
                if cols[1].button("Next →", disabled=st.session_state.viz_page >= total_pages - 1):
                    st.session_state.viz_page += 1
            
            # Show items for current page
            start_idx = st.session_state.viz_page * items_per_page
            end_idx = min(start_idx + items_per_page, len(items_to_show))
            
            for id, item in items_to_show[start_idx:end_idx]:
                st.sidebar.markdown(f"**{item['caption']}** (Page {item['page']})")
                
                if item["type"] == "figure":
                    try:
                        image = Image.open(io.BytesIO(item["content"]))
                        st.sidebar.image(image, caption=item["caption"], use_column_width=True)
                    except Exception as e:
                        st.sidebar.warning(f"Could not display image: {e}")
                else:  # Table
                    st.sidebar.code(item["content"])
                
                st.sidebar.markdown("---")

# Main content area based on mode
main_col1, main_col2 = st.columns([2, 1])

with main_col1:
    # Main content area - QA Interface
    st.markdown("<h3 class='subheader'>Ask Questions About Your PDFs</h3>", unsafe_allow_html=True)
    
    # Display suggested questions
    if st.session_state.suggested_questions:
        st.write("**Suggested Questions:**")
        
        # Use smaller columns for better layout
        cols = st.columns(2)
        for i, question in enumerate(st.session_state.suggested_questions):
            col_idx = i % 2
            with cols[col_idx]:
                if st.button(f"📝 {question}", key=f"suggest_{i}_{st.session_state.cache_key}", 
                         help="Click to ask this question", use_container_width=True):
                    use_suggested_question(question)
    
    # Question input area with voice option
    st.markdown("**Ask a question:**")
    input_col1, input_col2 = st.columns([5, 1])
    
    with input_col1:
        question = st.text_input("", key="question_input", placeholder="Type your question here...")
    
    with input_col2:
        voice_button = st.button("🎤 Voice", help="Ask question using voice")
        if voice_button:
            spoken_text = listen_for_question()
            if spoken_text:
                st.session_state.question_input = spoken_text
                st.session_state.process_question = True
                st.session_state.question_to_process = spoken_text
                st.rerun()
    
    ask_button = st.button("Ask", type="primary", use_container_width=True)
    
    # Process questions either from button press or from suggested questions
    if uploaded_files and collection and ((ask_button and question) or 
                                       (st.session_state.process_question and st.session_state.question_to_process)):
        # Use the question from session state if it's a suggested question
        question_to_process = st.session_state.question_to_process if st.session_state.process_question else question
        
        # Only process if it's a new question
        if not st.session_state.qa_history or question_to_process != st.session_state.last_question:
            process_query(question_to_process, collection)
        
        # Reset the process flag
        st.session_state.process_question = False
        st.session_state.question_to_process = ""

with main_col2:
    # Right column for learning progress and stats
    if uploaded_files:
        # Show document statistics
        st.markdown("<h3 class='subheader'>Document Statistics</h3>", unsafe_allow_html=True)
        
        total_pages = sum(meta["pages"] for meta in st.session_state.pdf_metadata.values())
        total_words = sum(meta["word_count"] for meta in st.session_state.pdf_metadata.values())
        
        stats_cols = st.columns(3)
        stats_cols[0].metric("Files", len(st.session_state.pdf_metadata))
        stats_cols[1].metric("Pages", total_pages)
        stats_cols[2].metric("Words", f"{total_words:,}")
        
        # Word cloud of topics
        if st.session_state.extracted_topics:
            st.markdown("<h3 class='subheader'>Topic Cloud</h3>", unsafe_allow_html=True)
            
            # Create word counts for topics based on passage counts
            topic_weights = {topic: max(5, len(passages)) for topic, passages in st.session_state.extracted_topics.items()}
            
            # Generate word cloud
            try:
                wc = WordCloud(width=400, height=200, background_color="white", 
                              contour_width=1, contour_color='steelblue', 
                              max_words=50)
                wc.generate_from_frequencies(topic_weights)
                
                # Display word cloud
                plt.figure(figsize=(10, 5))
                plt.imshow(wc, interpolation='bilinear')
                plt.axis("off")
                plt.tight_layout(pad=0)
                st.pyplot(plt)
            except Exception as e:
                st.warning(f"Could not generate topic cloud: {e}")
        
        # Learning progress
        if st.session_state.learning_progress:
            st.markdown("<h3 class='subheader'>Learning Progress</h3>", unsafe_allow_html=True)
            
            # Calculate overall progress
            if st.session_state.learning_progress:
                overall_progress = sum(st.session_state.learning_progress.values()) / len(st.session_state.learning_progress)
                st.progress(overall_progress / 100)
                
                # Show which topics need more attention
                low_progress_topics = [topic for topic, progress in st.session_state.learning_progress.items() 
                                     if progress < 50]
                
                if low_progress_topics:
                    st.markdown("**Explore these topics more:**")
                    for topic in low_progress_topics[:3]:
                        if st.button(f"Learn about '{topic}'", key=f"learn_{topic}"):
                            # Generate a question about this topic
                            topic_question = f"Explain the concept of {topic} in detail"
                            use_suggested_question(topic_question)

# Display current quiz if available
if st.session_state.current_quiz:
    st.markdown("<h3 class='subheader'>Knowledge Quiz</h3>", unsafe_allow_html=True)
    
    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}
    
    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False
    
    quiz_data = st.session_state.current_quiz
    
    if not st.session_state.quiz_submitted:
        # Display quiz questions
        for i, q in enumerate(quiz_data):
            st.markdown(f"**Question {i+1}:** {q['question']}")
            
            # Radio buttons for options
            selected = st.radio(
                f"Select answer for question {i+1}:",
                options=q['options'],
                key=f"quiz_q{i}"
            )
            
            # Store selected answer
            selected_idx = q['options'].index(selected) if selected in q['options'] else -1
            st.session_state.quiz_answers[i] = selected_idx
            
            st.markdown("---")
        
        # Submit button
        if st.button("Submit Quiz"):
            st.session_state.quiz_submitted = True
            st.rerun()
    else:
        # Show quiz results
        correct_count = 0
        
        for i, q in enumerate(quiz_data):
            user_answer_idx = st.session_state.quiz_answers.get(i, -1)
            correct_idx = q['correctAnswerIndex']
            
            is_correct = user_answer_idx == correct_idx
            if is_correct:
                correct_count += 1
            
            st.markdown(f"**Question {i+1}:** {q['question']}")
            
            # Show all options with formatting
            for j, option in enumerate(q['options']):
                prefix = "✓ " if j == correct_idx else "  "
                if j == user_answer_idx and j != correct_idx:
                    st.markdown(f"❌ {option}")
                elif j == correct_idx:
                    st.markdown(f"✅ {option}")
                else:
                    st.markdown(f"   {option}")
            
            # Show explanation
            st.info(f"**Explanation:** {q['explanation']}")
            st.markdown("---")
        
        # Show score
        score_percentage = (correct_count / len(quiz_data)) * 100
        st.success(f"Your score: {correct_count}/{len(quiz_data)} ({score_percentage:.1f}%)")
        
        # Button to reset quiz
        if st.button("Try Another Quiz"):
            st.session_state.quiz_submitted = False
            st.session_state.current_quiz = []
            st.session_state.quiz_answers = {}
            st.rerun()

# Display Q&A history
if st.session_state.qa_history:
    st.markdown("<h3 class='subheader'>Q&A History</h3>", unsafe_allow_html=True)
    
    for idx, (q, a, citations) in enumerate(st.session_state.qa_history[::-1]):
        real_idx = len(st.session_state.qa_history) - 1 - idx
        
        with st.expander(f"Q: {q}", expanded=(idx == 0)):
            st.markdown(f"**Answer:**")
            st.markdown(a)
            
            # Check if we need to create visualization
            viz_fig = create_visualization(a)
            if viz_fig:
                st.pyplot(viz_fig)
            
            # Show citations if available
            if citations:
                st.markdown("**Sources:**")
                for citation in citations:
                    st.markdown(f"""
                    <div class="citation">
                        <strong>Source:</strong> {citation['file']}, Page {citation['page']} <br/>
                        <strong>Relevance:</strong> {citation['relevance']:.2f} <br/>
                        <strong>Preview:</strong> <em>{citation['preview']}</em>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Button row
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button(f"🔊 Read Aloud", key=f"read_{real_idx}_{st.session_state.cache_key}"):
                    read_answer(real_idx)

# Export Q&A
if st.session_state.qa_history:
    st.markdown("---")
    export_col1, export_col2 = st.columns([1, 1])
    
    with export_col1:
        if st.button("📥 Download Q&A History"):
            qa_text = "\n\n".join([f"Q: {q}\n\nA: {a}" for q, a, _ in st.session_state.qa_history])
            filename = f"qa_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            b64 = base64.b64encode(qa_text.encode()).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Click here to download your Q&A</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    with export_col2:
        if st.button("📝 Generate Study Notes"):
            # Create structured study notes from Q&A history
            with st.spinner("Generating study notes..."):
                qa_pairs = [f"## Question: {q}\n\n{a}" for q, a, _ in st.session_state.qa_history]
                qa_content = "\n\n".join(qa_pairs)
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert at organizing information into clear, structured study notes. Create well-formatted study notes based on these question-answer pairs."},
                        {"role": "user", "content": f"Please organize these Q&A pairs into comprehensive study notes:\n\n{qa_content}"}
                    ],
                    max_tokens=1500,
                    temperature=0.3
                )
                notes = response["choices"][0]["message"]["content"]
                
                # Create downloadable file
                filename = f"study_notes_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                b64 = base64.b64encode(notes.encode()).decode()
                href = f'<a href="data:text/markdown;base64,{b64}" download="{filename}">Click here to download your study notes</a>'
                st.markdown(href, unsafe_allow_html=True)
            