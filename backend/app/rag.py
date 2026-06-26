import fitz  # PyMuPDF
import faiss
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# In-memory storage
chunks = []
index = None

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file"""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    result = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            result.append(chunk)
    return result

def index_document(pdf_path: str):
    """Process PDF and add to FAISS index"""
    global chunks, index

    text = extract_text_from_pdf(pdf_path)
    new_chunks = chunk_text(text)
    chunks.extend(new_chunks)

    embeddings = embedder.encode(new_chunks, convert_to_numpy=True)

    if index is None:
        index = faiss.IndexFlatL2(embeddings.shape[1])

    index.add(embeddings)
    return len(new_chunks)

def search_chunks(query: str, top_k: int = 5) -> list[str]:
    """Find most relevant chunks for a query"""
    if index is None or len(chunks) == 0:
        return []

    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    return [chunks[i] for i in indices[0] if i < len(chunks)]

def ask_llm(query: str, context_chunks: list[str]) -> str:
    """Send query + context to LLaMA and get answer"""
    context = "\n\n".join(context_chunks)

    prompt = f"""You are a helpful manufacturing assistant. 
Use the following SOP document excerpts to answer the question.
Only answer based on the provided context.

Context:
{context}

Question: {query}

Answer:"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

def query_rag(question: str) -> str:
    """Main RAG function — search + answer"""
    relevant_chunks = search_chunks(question)
    if not relevant_chunks:
        return "No relevant information found in the uploaded documents."
    return ask_llm(question, relevant_chunks)