import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textwrap, re, html
from typing import List
import numpy as np

# --- Page config and CSS ---
st.set_page_config(page_title="RAG Study Assistant", page_icon="📘", layout="wide")

st.markdown(
    """
    <style>
    /* Theme colors */
    :root {
        --accent: #0b7bff;
        --card-bg: #ffffff;
        --muted: #6b7280;
        --highlight: #fff176; /* yellow marker */
    }
    body {
        background-color: #f4f7fb;
        color: #111827;
        font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }
    .top-search {
        max-width:900px;
        margin: 10px auto 20px;
    }
    .search-box {
        border-radius: 14px;
        box-shadow: 0 6px 18px rgba(14,30,37,0.08);
        padding: 16px;
        background: linear-gradient(90deg, rgba(255,255,255,0.9), rgba(250,250,255,0.9));
        display:flex;
        align-items:center;
    }
    .stTextInput>div>div>input {
        height:44px;
        font-size:16px;
    }
    .card {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 6px 18px rgba(14,30,37,0.06);
        margin-bottom: 18px;
    }
    .meta {
        color: var(--muted);
        font-size:13px;
        margin-bottom:8px;
    }
    mark.rag-highlight {
        background: var(--highlight);
        padding: 0 4px;
        border-radius: 3px;
    }
    .pdf-list { font-size:14px; color:#0b7bff; margin-bottom:8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Session state initialization ---
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks: List[str] = []
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names: List[str] = []

# --- Helpers ---
def split_text(text: str, chunk_size: int = 300) -> List[str]:
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def clean_text(text: str) -> str:
    # basic cleaning
    text = text.replace("�", " ")
    text = re.sub(r'\s+\n', '\n', text)  # remove spaces before newlines
    text = re.sub(r'\n{3,}', '\n\n', text)  # collapse many newlines
    return text.strip()

def wrap_text(text: str, width: int = 110) -> str:
    lines = text.split("\n")
    wrapped = "\n".join(textwrap.fill(line, width=width) if len(line) > width else line for line in lines)
    return wrapped

def highlight_terms(text: str, query: str) -> str:
    """
    Highlight query terms in text using <mark class='rag-highlight'>...</mark>.
    Case-insensitive. Escapes HTML in the text to be safe.
    """
    # escape HTML first
    safe_text = html.escape(text)

    # build term list from query (words longer than 1 char)
    terms = re.findall(r'\w+', query)
    terms = [t for t in terms if len(t) > 1]
    if not terms:
        return safe_text

    # sort by length desc to avoid partial overlaps (e.g., "data" before "at")
    terms = sorted(set(terms), key=lambda s: -len(s))

    # for each term, replace occurrences with highlighted HTML (case-insensitive)
    for term in terms:
        # create regex, use word boundaries for cleaner highlights
        pattern = re.compile(r"(?i)\b(" + re.escape(term) + r")\b")
        safe_text = pattern.sub(r"<mark class='rag-highlight'>\1</mark>", safe_text)

    return safe_text

def process_uploaded_files(files):
    st.session_state.document_chunks = []
    st.session_state.uploaded_names = []
    for file in files:
        reader = PdfReader(file)
        doc_text = ""
        for p in reader.pages:
            text = p.extract_text()
            if text:
                doc_text += text + "\n"
        doc_text = clean_text(doc_text)
        if doc_text:
            chunks = split_text(doc_text, chunk_size=280)
            st.session_state.document_chunks.extend(chunks)
            st.session_state.uploaded_names.append(getattr(file, "name", "uploaded"))
    # build vector index
    if st.session_state.document_chunks:
        st.session_state.vectorizer = TfidfVectorizer()
        st.session_state.vectors = st.session_state.vectorizer.fit_transform(st.session_state.document_chunks)

def get_top_k_answers(query: str, k: int = 3):
    if st.session_state.vectors is None:
        return []
    q_vec = st.session_state.vectorizer.transform([query])
    sims = cosine_similarity(q_vec, st.session_state.vectors)[0]
    # get indices of top k (descending)
    top_k_idx = np.argsort(sims)[::-1][:k]
    results = []
    for idx in top_k_idx:
        results.append({"index": int(idx), "score": float(sims[idx]), "text": st.session_state.document_chunks[idx]})
    return results

# --- Layout: Sidebar (left) and main content ---
sidebar = st.sidebar
sidebar.title("📂 Documents & Controls")

uploaded = sidebar.file_uploader("Upload PDFs (multiple)", accept_multiple_files=True, type=["pdf"])
if uploaded:
    if st.sidebar.button("Process uploaded PDFs"):
        process_uploaded_files(uploaded)
        st.sidebar.success(f"Processed {len(st.session_state.document_chunks)} chunks from uploaded PDFs")
else:
    sidebar.info("Upload PDFs and click 'Process uploaded PDFs'")

if st.session_state.uploaded_names:
    sidebar.markdown("**Uploaded files:**")
    for n in st.session_state.uploaded_names:
        sidebar.markdown(f"- <span class='pdf-list'>{n}</span>", unsafe_allow_html=True)

sidebar.markdown("---")
sidebar.markdown("**Settings**")
k_choice = sidebar.slider("Number of results", min_value=1, max_value=5, value=3, step=1)
sidebar.markdown("")

# --- Top-center Search box ---
st.markdown('<div class="top-search">', unsafe_allow_html=True)
with st.container():
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.markdown('<div class="search-box">', unsafe_allow_html=True)
        query = st.text_input("🔎 Ask a question from the uploaded PDFs", placeholder="Type your question and press Enter...", key="top_query")
        st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Main area: Show results when query is provided ---
st.write("")  # spacing
if query and query.strip():
    if st.session_state.vectors is None or not st.session_state.document_chunks:
        st.warning("No PDFs processed yet. Upload and process PDFs from the sidebar.")
    else:
        # compute top-k
        top_results = get_top_k_answers(query, k=k_choice)
        st.markdown(f"### 🔎 Results for: _{html.escape(query)}_")
        st.markdown("Tip: terms from your question are highlighted in yellow in the results.\n")
        # display each result in a card
        for i, res in enumerate(top_results, start=1):
            score_pct = round(res["score"] * 100, 2)
            # prepare highlighted & wrapped text
            raw_text = clean_text(res["text"])
            wrapped = wrap_text(raw_text, width=120)
            highlighted = highlight_terms(wrapped, query)

            st.markdown(
                f"""
                <div class="card">
                    <div class="meta">Result #{i} — Similarity: {score_pct}%</div>
                    <div style="font-size:15px; line-height:1.7;">{highlighted}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # convenience: show a small "export answers" area
        st.markdown("---")
        if st.button("Copy all answers to clipboard (browser)"):
            # combine top results into one string
            combined = "\n\n".join([clean_text(r["text"]) for r in top_results])
            st.write("Select the text below and copy manually (browser clipboard from Python may be restricted):")
            st.code(combined)

# Footer / small help
st.markdown("---")
st.markdown("Made with ❤️ — Local RAG Study Assistant. Upload PDFs, process them, then ask questions from the top search box.")


 

