import fitz  # PyMuPDF
import faiss
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from pathlib import Path

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# In-memory storage
documents = []
index = None
bm25 = None

# ── Hierarchical Chunking ────────────────────────────────────────────

def extract_sections(page_text: str, page_num: int, doc_name: str) -> list[dict]:
    chunks = []
    current_section = "General"
    lines = page_text.split('\n')
    buffer = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        is_header = (
            (line.isupper() and len(line) > 3) or
            (len(line) < 80 and line[0].isdigit() and '.' in line[:3])
        )

        if is_header:
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
            chunks.append({
                "text": line,
                "page": page_num,
                "section": line,
                "doc_name": doc_name,
                "level": "section"
            })
        else:
            buffer.append(line)
            if len(buffer) >= 8:
                chunk_text = " ".join(buffer)
                chunks.append({
                    "text": chunk_text,
                    "page": page_num,
                    "section": current_section,
                    "doc_name": doc_name,
                    "level": "paragraph"
                })
                buffer = buffer[-2:]

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

    texts = [d["text"] for d in documents]
    embeddings = embedder.encode(texts, convert_to_numpy=True)

    if index is None:
        index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    return len(new_docs)

# ── Hybrid Search ─────────────────────────────────────────────────────

def search_chunks(query: str, top_k: int = 5) -> list[dict]:
    if index is None or len(documents) == 0:
        return []

    # FAISS semantic search
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, faiss_indices = index.search(query_embedding, top_k * 2)
    faiss_scores = {int(i): 1 / (1 + d) for i, d in zip(faiss_indices[0], distances[0]) if i < len(documents)}

    # BM25 keyword search
    tokenized_query = query.lower().split()
    bm25_scores_raw = bm25.get_scores(tokenized_query)
    bm25_top = np.argsort(bm25_scores_raw)[::-1][:top_k * 2]
    max_bm25 = max(bm25_scores_raw) if max(bm25_scores_raw) > 0 else 1
    bm25_scores = {int(i): bm25_scores_raw[i] / max_bm25 for i in bm25_top}

    # Combine scores (60% semantic, 40% keyword)
    all_indices = set(faiss_scores.keys()) | set(bm25_scores.keys())
    combined = {}
    for i in all_indices:
        semantic = faiss_scores.get(i, 0)
        keyword = bm25_scores.get(i, 0)
        combined[i] = 0.6 * semantic + 0.4 * keyword

    top_indices = sorted(combined.keys(), key=lambda x: combined[x], reverse=True)[:top_k]
    return [documents[i] for i in top_indices if i < len(documents)]

# ── Answer Generation with Citations ─────────────────────────────────

def ask_llm(query: str, context_docs: list[dict]) -> dict:
    context_parts = []
    for doc in context_docs:
        context_parts.append(
            f"[Source: {doc['doc_name']}, Page {doc['page']}, Section: {doc['section']}]\n{doc['text']}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""You are a precise manufacturing SOP assistant.
Answer the question using ONLY the provided context.
Do not include any source references or citations in your answer.
Give a direct, concise answer only.

Context:
{context}

Question: {query}

Answer:"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response["message"]["content"]

    # Extract unique citations separately
    citations = list(set([
        f"{d['doc_name']} — Page {d['page']}, Section: {d['section']}"
        for d in context_docs
    ]))

    return {"answer": answer, "citations": citations}

def query_rag(question: str) -> str:
    relevant_docs = search_chunks(question)
    if not relevant_docs:
        return "No relevant information found in the uploaded documents."
    result = ask_llm(question, relevant_docs)
    answer = result["answer"]
    if result["citations"]:
        answer += "\n\nSources:\n" + "\n".join([f"- {c}" for c in result["citations"]])
    return answer