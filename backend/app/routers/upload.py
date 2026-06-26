from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import shutil
from app.rag import index_document

router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and index a PDF SOP document"""

    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Save file
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Index document into RAG
    num_chunks = index_document(str(file_path))

    return {
        "message": f"Successfully uploaded and indexed {file.filename}",
        "filename": file.filename,
        "chunks_indexed": num_chunks
    }

@router.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    files = list(UPLOAD_DIR.glob("*.pdf"))
    return {
        "documents": [f.name for f in files],
        "total": len(files)
    }