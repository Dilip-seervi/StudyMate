import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import textwrap, re, html
from typing import List
import numpy as np

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="RAG Study Assistant (FAISS)",
    page_icon="📘",
    layout="wide"
)

# ---------------- CSS Styling ----------------
st.markdown(
    """
    <style>
    body { background-color: #f4f7fb; }
    .card {
        background: white;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 6px 18px rgba(14,30,37,0.06);
        margin-bottom: 20px;
    }
    .meta { color: #6b7280; font-size: 13px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Session State ----------------
if "chunks" not in st.session_state:
    st.session_state.chunks: List[str] = []

if "index" not in st.session_state:
    st.session_state.index = None

if "model" not in st.session_state:
    st.session_state.model = SentenceTransformer("all-MiniLM-L6-v2")

if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = []

# ---------------- Helpers ----------------
def clean_text(text: str) -> str:
    text = text.replace("�", " ")
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_text(text: str, sentences_per_chunk=4):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [
        " ".join(sentences[i:i + sentences_per_chunk])
        for i in range(0, len(sentences), sentences_per_chunk)
        if len(" ".join(sentences[i:i + sentences_per_chunk])) > 40
    ]

def wrap_text(text, width=120):
    return "\n".join(textwrap.fill(text, width=width).split("\n"))

def process_pdfs(files):
    st.session_state.chunks = []
    st.session_state.uploaded_names = []

    for file in files:
        reader = PdfReader(file)
        full_text = ""

        for page in reader.pages:
            if page.extract_text():
                full_text += page.extract_text() + "\n"

        full_text = clean_text(full_text)
        chunks = split_text(full_text)
        st.session_state.chunks.extend(chunks)
        st.session_state.uploaded_names.append(file.name)

    embeddings = st.session_state.model.encode(
        st.session_state.chunks,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    st.session_state.index = index

def search(query, k):
    q_emb = st.session_state.model.encode([query])
    distances, indices = st.session_state.index.search(q_emb, k)
    return indices[0], distances[0]

# ---------------- Sidebar ----------------
sidebar = st.sidebar
sidebar.title("📂 Documents")

files = sidebar.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if files and sidebar.button("Process PDFs"):
    process_pdfs(files)
    sidebar.success("PDFs processed with FAISS")

k_choice = sidebar.slider("Number of answers", 1, 5, 3)

# ---------------- Search ----------------
query = st.text_input("🔎 Ask an exam question")

if query and st.session_state.index:
    indices, distances = search(query, k_choice)

    st.markdown(f"### 📘 Results for: *{html.escape(query)}*")

    for rank, (idx, dist) in enumerate(zip(indices, distances), start=1):
        text = wrap_text(clean_text(st.session_state.chunks[idx]))
        score = round(100 / (1 + dist), 2)

        st.markdown(
            f"""
            <div class="card">
                <div class="meta">Answer {rank} — Relevance: {score}%</div>
                <div style="white-space: pre-wrap; font-size: 15px;">
                    {html.escape(text)}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------- Footer ----------------
st.markdown("---")
st.markdown("📘 **FAISS-powered Exam RAG — Separate Answers Mode**")
