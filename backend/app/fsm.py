import re
from enum import Enum

class StepStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"

class FSMState:
    def __init__(self, step_id: int, text: str):
        self.step_id = step_id
        self.text = text
        self.status = StepStatus.PENDING
        self.transitions = {
            "next": step_id + 1,
            "on_fail": None,
            "on_repeat": None,
            "on_skip": step_id + 1,
        }
        self.is_conditional = False
        self.condition_text = None
        self.branches = {}

CONDITIONAL_PATTERNS = [
    (r'\bif\b.*\bproceed\b', 'branch'),
    (r'\bif\b.*\badjust\b', 'branch'),
    (r'\bif\b.*\brepeat\b', 'loop'),
    (r'\botherwise\b', 'branch'),
    (r'\bif not\b', 'branch'),
    (r'\bif.*deviat', 'branch'),
    (r'\brepeat steps?\b', 'loop'),
    (r'\bdo not use\b', 'terminal'),
    (r'\bstop\b.*\bcontact\b', 'terminal'),
    (r'\bcontact.*maintenance\b', 'terminal'),
]

def detect_conditional(text: str) -> tuple[bool, str | None]:
    text_lower = text.lower()
    for pattern, branch_type in CONDITIONAL_PATTERNS:
        if re.search(pattern, text_lower):
            return True, branch_type
    return False, None

def build_fsm(steps: list[str]) -> list[FSMState]:
    states = []

    for i, step_text in enumerate(steps):
        state = FSMState(step_id=i, text=step_text)
        is_conditional, branch_type = detect_conditional(step_text)

        if is_conditional:
            state.is_conditional = True
            state.condition_text = branch_type

            if branch_type == 'loop':
                loop_match = re.search(r'repeat steps?\s+(\d+)', step_text.lower())
                if loop_match:
                    loop_to = int(loop_match.group(1)) - 1
                    state.transitions["on_repeat"] = loop_to
                    state.branches = {"pass": i + 1, "fail": loop_to}
                else:
                    state.branches = {"pass": i + 1, "fail": max(0, i - 2)}

            elif branch_type == 'terminal':
                state.transitions["on_fail"] = -1
                state.branches = {"pass": i + 1, "fail": -1}

            else:
                state.branches = {"pass": i + 1, "fail": i}

        else:
            state.branches = {"pass": i + 1, "fail": i}

        states.append(state)

    return states

class FSMSession:
    def __init__(self, session_id: str, steps: list[str]):
        self.session_id = session_id
        self.states = build_fsm(steps)
        self.current_step = 0
        self.history = []
        self.completed = False
        self.terminated = False

    def get_current_state(self) -> dict:
        if self.current_step >= len(self.states):
            self.completed = True
            return {"completed": True, "message": "All steps completed successfully."}

        state = self.states[self.current_step]
        return {
            "step_id": state.step_id,
            "step_number": state.step_id + 1,
            "text": state.text,
            "is_conditional": state.is_conditional,
            "condition_type": state.condition_text,
            "branches": state.branches,
            "total_steps": len(self.states),
            "completed": False,
            "terminated": False
        }

    def advance(self, outcome: str = "pass") -> dict:
        if self.current_step >= len(self.states):
            return {"completed": True}

        state = self.states[self.current_step]
        self.history.append({
            "step_id": self.current_step,
            "outcome": outcome,
            "text": state.text
        })

        next_step = state.branches.get(outcome, self.current_step)

        if next_step == -1:
            self.terminated = True
            return {
                "terminated": True,
                "message": "Process terminated. Please contact the maintenance team immediately.",
                "history": self.history
            }

        self.current_step = next_step
        return self.get_current_state()

    def get_progress(self) -> dict:
        return {
            "current_step": self.current_step + 1,
            "total_steps": len(self.states),
            "history": self.history,
            "completed": self.completed,
            "terminated": self.terminated,
            "percentage": round((self.current_step / len(self.states)) * 100)
        }

fsm_sessions: dict[str, FSMSession] = {}

def create_fsm_session(session_id: str, steps: list[str]) -> dict:
    session = FSMSession(session_id, steps)
    fsm_sessions[session_id] = session
    return session.get_current_state()

def advance_fsm_session(session_id: str, outcome: str = "pass") -> dict:
    if session_id not in fsm_sessions:
        return {"error": "Session not found"}
    return fsm_sessions[session_id].advance(outcome)

def get_fsm_progress(session_id: str) -> dict:
    if session_id not in fsm_sessions:
        return {"error": "Session not found"}
    return fsm_sessions[session_id].get_progress()

def get_fsm_session(session_id: str) -> dict:
    if session_id not in fsm_sessions:
        return {"error": "Session not found"}
    return fsm_sessions[session_id].get_current_state()