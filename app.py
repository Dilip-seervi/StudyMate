import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="RAG Study App", page_icon="📚")

# Session state to hold processed documents
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []

if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None

if "vectors" not in st.session_state:
    st.session_state.vectors = None


def split_text(text, chunk_size=200):
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks


st.title("📚 Local RAG Study App (No GPU/Internet Needed)")

menu = st.sidebar.selectbox("Menu", ["Upload PDFs", "Ask Questions"])

if menu == "Upload PDFs":
    uploaded_files = st.file_uploader("Upload study PDFs", accept_multiple_files=True)

    if uploaded_files:
        st.session_state.document_chunks.clear()

        for file in uploaded_files:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
            st.session_state.document_chunks.extend(split_text(text))

        st.session_state.vectorizer = TfidfVectorizer()
        st.session_state.vectors = st.session_state.vectorizer.fit_transform(st.session_state.document_chunks)

        st.success("PDFs processed successfully!")

elif menu == "Ask Questions":
    if not st.session_state.document_chunks:
        st.warning("Upload PDFs first!")
    else:
        query = st.text_input("Ask a question:")
        if query:
            q_vec = st.session_state.vectorizer.transform([query])
            similarity = cosine_similarity(q_vec, st.session_state.vectors)[0]
            top_idx = similarity.argmax()
            answer = st.session_state.document_chunks[top_idx]

            st.subheader("📌 Best Match from PDF:")
            st.write(answer)

