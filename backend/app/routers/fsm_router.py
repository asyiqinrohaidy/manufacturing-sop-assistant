from fastapi import APIRouter
from pydantic import BaseModel
from app.fsm import create_fsm_session, advance_fsm_session, get_fsm_progress, get_fsm_session
from app.rag import search_chunks
from app.guided import get_guided_response
import uuid

router = APIRouter()

class StartFSMRequest(BaseModel):
    question: str

class AdvanceFSMRequest(BaseModel):
    session_id: str
    outcome: str = "pass"  # "pass" or "fail"

@router.post("/fsm/start")
async def start_fsm_session(request: StartFSMRequest):
    """Start a new FSM guided session"""
    # Get steps from RAG
    context_docs = search_chunks(request.question)
    if not context_docs:
        return {"error": "No relevant SOP found for this question"}

    guided = get_guided_response(request.question, context_docs)
    steps = guided["steps"]

    # Create FSM session
    session_id = str(uuid.uuid4())
    current_state = create_fsm_session(session_id, steps)

    return {
        "session_id": session_id,
        "total_steps": len(steps),
        "all_steps": steps,
        "current_state": current_state
    }

@router.post("/fsm/advance")
async def advance_session(request: AdvanceFSMRequest):
    """Advance FSM to next step"""
    result = advance_fsm_session(request.session_id, request.outcome)
    return result

@router.get("/fsm/progress/{session_id}")
async def get_progress(session_id: str):
    """Get FSM session progress"""
    return get_fsm_progress(session_id)

@router.get("/fsm/state/{session_id}")
async def get_state(session_id: str):
    """Get current FSM state"""
    return get_fsm_session(session_id)