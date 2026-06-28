import fitz  # PyMuPDF
import faiss
import numpy as np
import ollama
import time
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from pathlib import Path

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# In-memory storage
documents = []  # {"text": str, "page": int, "section": str, "doc_name": str, "level": str}
index = None
bm25 = None

# ── Hierarchical Chunking ────────────────────────────────────────────

def extract_sections(page_text: str, page_num: int, doc_name: str) -> list[dict]:
    """Extract sections and paragraphs from a page — hierarchical chunking"""
    chunks = []
    current_section = "General"
    lines = page_text.split('\n')
    buffer = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect section headers (ALL CAPS or numbered like "1.", "2.")
        is_header = (
            (line.isupper() and len(line) > 3) or
            (len(line) < 80 and line[0].isdigit() and '.' in line[:3])
        )

        if is_header:
            # Save previous buffer as a chunk
            if buffer:
                chunk_text = " ".join(buffer)
                if len(chunk_text) > 50:
                    chunks.append({
                        "text": chunk_text,
                        "page": page_num,
                        "section": current_section,
                        "doc_name": doc_name,
                        "level": "paragraph"
                    })
            current_section = line
            buffer = []
            # Add section header itself as a chunk
            chunks.append({
                "text": line,
                "page": page_num,
                "section": line,
                "doc_name": doc_name,
                "level": "section"
            })
        else:
            buffer.append(line)
            # Split into paragraph chunks every 8 lines
            if len(buffer) >= 8:
                chunk_text = " ".join(buffer)
                chunks.append({
                    "text": chunk_text,
                    "page": page_num,
                    "section": current_section,
                    "doc_name": doc_name,
                    "level": "paragraph"
                })
                buffer = buffer[-2:]  # Keep overlap

    # Save remaining buffer
    if buffer:
        chunk_text = " ".join(buffer)
        if len(chunk_text) > 50:
            chunks.append({
                "text": chunk_text,
                "page": page_num,
                "section": current_section,
                "doc_name": doc_name,
                "level": "paragraph"
            })

    return chunks

def index_document(pdf_path: str) -> int:
    """Process PDF with hierarchical chunking and index"""
    global documents, index, bm25

    doc = fitz.open(pdf_path)
    doc_name = Path(pdf_path).stem
    new_docs = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            page_chunks = extract_sections(text, page_num, doc_name)
            new_docs.extend(page_chunks)

    documents.extend(new_docs)

    # Build FAISS index
    texts = [d["text"] for d in documents]
    embeddings = embedder.encode(texts, convert_to_numpy=True)

    if index is None:
        index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Build BM25 index
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    return len(new_docs)

# ── HyDE Query Rewriting ─────────────────────────────────────────────

def hyde_rewrite(query: str) -> str:
    """Generate hypothetical answer to improve retrieval"""
    prompt = f"""Generate a brief hypothetical answer (2-3 sentences) for this manufacturing SOP question. 
Be specific and technical. Do not say you don't know.

Question: {query}

Hypothetical answer:"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

# ── Hybrid Search ─────────────────────────────────────────────────────

def search_chunks(query: str, top_k: int = 5) -> list[dict]:
    """Hybrid search: combine FAISS semantic + BM25 keyword results"""
    if index is None or len(documents) == 0:
        return []

    # Skip HyDE for local setup — use query directly
    combined_query = query

    # FAISS semantic search
    query_embedding = embedder.encode([combined_query], convert_to_numpy=True)
    distances, faiss_indices = index.search(query_embedding, top_k * 2)
    faiss_scores = {int(i): 1 / (1 + d) for i, d in zip(faiss_indices[0], distances[0]) if i < len(documents)}

    # BM25 keyword search
    tokenized_query = combined_query.lower().split()
    bm25_scores_raw = bm25.get_scores(tokenized_query)
    bm25_top = np.argsort(bm25_scores_raw)[::-1][:top_k * 2]
    max_bm25 = max(bm25_scores_raw) if max(bm25_scores_raw) > 0 else 1
    bm25_scores = {int(i): bm25_scores_raw[i] / max_bm25 for i in bm25_top}

    # Combine scores (weighted: 60% semantic, 40% keyword)
    all_indices = set(faiss_scores.keys()) | set(bm25_scores.keys())
    combined = {}
    for i in all_indices:
        semantic = faiss_scores.get(i, 0)
        keyword = bm25_scores.get(i, 0)
        combined[i] = 0.6 * semantic + 0.4 * keyword

    # Return top_k results
    top_indices = sorted(combined.keys(), key=lambda x: combined[x], reverse=True)[:top_k]
    return [documents[i] for i in top_indices if i < len(documents)]

# ── Answer Generation with Citations ─────────────────────────────────

def ask_llm(query: str, context_docs: list[dict]) -> dict:
    """Generate answer with page citations"""
    context_parts = []
    for doc in context_docs:
        context_parts.append(
            f"[Source: {doc['doc_name']}, Page {doc['page']}, Section: {doc['section']}]\n{doc['text']}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""You are a precise manufacturing SOP assistant.
Answer the question using ONLY the provided context.
At the end of your answer, list the sources you used.

Context:
{context}

Question: {query}

Answer (include sources at the end):"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response["message"]["content"]

    # Extract unique citations
    citations = list(set([
        f"{d['doc_name']} — Page {d['page']}, Section: {d['section']}"
        for d in context_docs
    ]))

    return {"answer": answer, "citations": citations}

def query_rag(question: str) -> str:
    """Main RAG function for regular Q&A"""
    relevant_docs = search_chunks(question)
    if not relevant_docs:
        return "No relevant information found in the uploaded documents."
    result = ask_llm(question, relevant_docs)
    answer = result["answer"]
    if result["citations"]:
        answer += "\n\n📄 Sources:\n" + "\n".join([f"• {c}" for c in result["citations"]])
    return answer