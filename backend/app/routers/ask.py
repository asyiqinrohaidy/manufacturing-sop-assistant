from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.rag import query_rag, search_chunks, ask_llm
from app.guided import is_procedural_question, get_guided_response

router = APIRouter()

class QuestionRequest(BaseModel):
    question: str

@router.post("/ask")
async def ask_question(request: QuestionRequest):
    """Ask a question about uploaded SOP documents"""

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    question = request.question

    # Check if procedural question — use Guided Mode
    if is_procedural_question(question):
        context_docs = search_chunks(question)

        if not context_docs:
            return {
                "type": "error",
                "answer": "No relevant information found in uploaded documents."
            }

        guided_response = get_guided_response(question, context_docs)

        # Add citations to guided response
        citations = list(set([
            f"{d['doc_name']} — Page {d['page']}, Section: {d['section']}"
            for d in context_docs
        ]))
        guided_response["citations"] = citations
        return guided_response

    # Regular Q&A with citations
    context_docs = search_chunks(question)
    if not context_docs:
        return {
            "type": "error",
            "answer": "No relevant information found in uploaded documents."
        }

    result = ask_llm(question, context_docs)
    return {
        "type": "regular",
        "answer": result["answer"],
        "citations": result["citations"]
    }