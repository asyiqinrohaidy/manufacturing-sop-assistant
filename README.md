# Manufacturing SOP Assistant

A RAG-powered assistant that enables factory operators to query Standard Operating Procedure (SOP) documents in natural language and receive interactive, step-by-step procedural guidance with compliance enforcement.

Built to address a real problem in manufacturing environments: operators spending significant time searching through lengthy SOP manuals during time-sensitive operations.

---

## Features

- **Natural Language Q&A** - Ask any question about uploaded SOP documents and receive accurate, context-grounded answers with exact page and section citations
- **FSM Guided Mode** - Procedural questions automatically trigger a Finite State Machine that delivers interactive step-by-step checklists with pass/fail branching
- **Compliance Enforcement** - Operators cannot skip a failed step; certain failure conditions trigger a terminal state requiring maintenance intervention
- **Hybrid Search** - Combines FAISS semantic search and BM25 keyword search (60/40 weighted) for more accurate retrieval than either method alone
- **Hierarchical Chunking** - Documents are parsed into section and paragraph-level chunks, preserving document structure for more precise retrieval
- **Page Citations** - Every answer includes exact source references (document name, page number, section) to support technical auditing
- **Multi-document Support** - Multiple SOP PDFs can be uploaded and queried simultaneously with cross-document retrieval

---

## Architecture

```
React Frontend (port 3000)
        |
        | HTTP (REST API)
        |
FastAPI Backend (port 8000)
        |
        |-- PDF Ingestion Pipeline
        |       |-- PyMuPDF (text extraction)
        |       |-- Hierarchical Chunking (section + paragraph level)
        |       |-- Sentence Transformers (embeddings)
        |       |-- FAISS (vector index)
        |       |-- BM25 (keyword index)
        |
        |-- Query Pipeline
        |       |-- Hybrid Search (FAISS + BM25)
        |       |-- LLaMA 3.2 via Ollama (answer generation)
        |       |-- Citation Extraction
        |
        |-- FSM Engine
                |-- Procedural Question Detection
                |-- State Graph Construction
                |-- Pass/Fail Branching
                |-- Terminal State Handling
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React.js |
| Backend | Python, FastAPI |
| LLM | LLaMA 3.2 via Ollama (local, no API cost) |
| Vector Search | FAISS |
| Keyword Search | BM25 (rank-bm25) |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| PDF Processing | PyMuPDF |
| Containerisation | Docker, Docker Compose |

---

## How It Works

### Regular Q&A Mode
User asks a factual question → hybrid search retrieves relevant chunks → LLaMA generates a grounded answer → citations returned with exact page and section references.

### FSM Guided Mode
User asks a procedural question (detected via keyword matching) → RAG retrieves relevant SOP sections → LLaMA extracts steps → FSM builds a state graph → operator navigates steps one at a time with pass/fail branching.

**FSM branching logic:**
- **Pass** - advance to next step
- **Fail (regular step)** - stay on current step, operator must retry
- **Fail (conditional step)** - branch to adjustment or loop step as defined in SOP
- **Fail (terminal step)** - process terminated, operator instructed to contact maintenance

---

## Setup

### Prerequisites
- Python 3.11
- Node.js 18+
- Ollama installed and running ([ollama.com](https://ollama.com))
- Docker and Docker Compose (optional)

### 1. Pull LLaMA model
```bash
ollama pull llama3.2
```

### 2. Backend setup
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 3. Frontend setup
```bash
cd frontend
npm install
npm start
```

### 4. Docker setup (configuration provided, not yet verified)
Dockerfile and docker-compose.yml are included for containerised deployment. This has not yet been tested end-to-end.
```bash
docker-compose up --build
```

---

## Usage

1. Open `http://localhost:3000`
2. Click **Upload SOP Document (PDF)** and upload your SOP file
3. Ask any question in the chat input:
   - Factual: *"What PPE is required before calibration?"*
   - Procedural: *"How to calibrate Tool A1 torque wrench?"*
4. For procedural questions, FSM Guided Mode activates automatically — click **Step Completed** or **Step Failed** to navigate

---

## Project Structure

```
manufacturing-sop-assistant/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI entry point
│   │   ├── rag.py           # RAG pipeline (chunking, hybrid search, LLM)
│   │   ├── guided.py        # Step extraction and formatting
│   │   ├── fsm.py           # Finite State Machine engine
│   │   └── routers/
│   │       ├── upload.py    # PDF upload endpoint
│   │       ├── ask.py       # Q&A endpoint
│   │       └── fsm_router.py # FSM endpoints
│   ├── uploads/             # Uploaded SOP documents
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.js           # Main React component
│   │   └── App.css          # Styling
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/upload` | Upload and index a PDF document |
| GET | `/api/documents` | List all uploaded documents |
| POST | `/api/ask` | Ask a question (Q&A or guided mode) |
| POST | `/api/fsm/start` | Start a new FSM guided session |
| POST | `/api/fsm/advance` | Advance FSM session (pass/fail) |
| GET | `/api/fsm/progress/{id}` | Get session progress |
| GET | `/api/fsm/state/{id}` | Get current FSM state |

---

## Design Decisions

**Why local LLM (Ollama) instead of OpenAI?**
Manufacturing environments often have data sensitivity requirements. Running LLaMA locally ensures SOP documents never leave the organisation's infrastructure, with zero per-query API cost.

**Why hybrid search instead of pure semantic search?**
SOP documents contain specific technical terms, tool names, and part numbers that semantic search alone may miss. BM25 keyword matching catches exact term matches while FAISS handles conceptual similarity. The 60/40 weighted combination outperforms either method alone on technical documents.

**Why FSM instead of a simple checklist?**
Real SOPs contain conditional logic, steps that branch based on measurement outcomes or error states. A simple linear checklist cannot enforce compliance for conditional procedures. The FSM models the actual decision tree embedded in the SOP.

---

## Future Improvements

- Voice input for hands-free factory floor operation
- Mobile responsive UI for tablet/handheld use
- Persistent session storage across server restarts
- User authentication and role-based access
- Support for additional document formats (Word, Excel)
- Multilingual support (Bahasa Malaysia, Mandarin)

---

## Author

Asyiqin Rohaidy,
AI Engineer
[LinkedIn](https://linkedin.com/in/asyiqinrohaidy) | [GitHub](https://github.com/asyiqinrohaidy)
