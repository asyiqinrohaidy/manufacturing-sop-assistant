import ollama

# Keywords yang indicate soalan prosedural
PROCEDURAL_KEYWORDS = [
    "how to", "how do", "steps to", "procedure for",
    "process of", "guide to", "instructions for",
    "how can i", "what are the steps", "walk me through"
]

def is_procedural_question(question: str) -> bool:
    """Detect if question is asking for a procedure/steps"""
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in PROCEDURAL_KEYWORDS)

def format_as_steps(answer: str) -> list[str]:
    """Extract steps from LLM answer and return as list"""
    lines = answer.strip().split("\n")
    steps = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove common numbering formats: "1.", "1)", "Step 1:", etc.
        import re
        cleaned = re.sub(r'^(step\s*)?\d+[\.\):\-]\s*', '', line, flags=re.IGNORECASE)
        cleaned = re.sub(r'^[\-\•\*]\s*', '', cleaned)

        if cleaned:
            steps.append(cleaned)

    return steps if steps else [answer]

def get_guided_response(question: str, context_chunks: list[str]) -> dict:
    """Generate guided step-by-step response"""
    context = "\n\n".join(context_chunks)

    prompt = f"""You are a manufacturing SOP assistant.
Based on the SOP document below, provide a clear step-by-step procedure.
Format your response as numbered steps only. Each step on a new line.
Start each line with the step number followed by a period.

Context:
{context}

Question: {question}

Steps:"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )

    raw_answer = response["message"]["content"]
    steps = format_as_steps(raw_answer)

    return {
        "type": "guided",
        "steps": steps,
        "total_steps": len(steps)
    }