import ollama
import re

PROCEDURAL_KEYWORDS = [
    "how to", "how do", "steps to", "procedure for",
    "process of", "guide to", "instructions for",
    "how can i", "what are the steps", "walk me through"
]

INTRO_PATTERNS = [
    r'^follow these steps',
    r'^here are the steps',
    r'^here is the procedure',
    r'^the following steps',
    r'^to calibrate',
    r'^based on the',
    r'^according to the',
    r'^sure',
    r'^certainly',
    r'^of course',
    r'^i can',
    r'^i will',
    r'^below are',
    r'^the steps',
    r'^this procedure',
]

def is_procedural_question(question: str) -> bool:
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in PROCEDURAL_KEYWORDS)

def format_as_steps(answer: str) -> list[str]:
    lines = answer.strip().split("\n")
    steps = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove "1." or "Step 1:" or "1)" formats
        cleaned = re.sub(r'^(step\s*)?\d+[\.\):\-]\s*', '', line, flags=re.IGNORECASE)
        # Remove double numbering like "1. 2."
        cleaned = re.sub(r'^\d+\.\s*\d+[\.\)]\s*', '', cleaned)
        # Remove leading bullet characters
        cleaned = re.sub(r'^[\-\*\•]\s*', '', cleaned)
        # Remove remaining leading numbers from original SOP
        cleaned = re.sub(r'^\d+\.\s+', '', cleaned)
        cleaned = cleaned.strip()

        if not cleaned or len(cleaned) < 10:
            continue

        # Filter out intro/preamble sentences
        is_intro = any(re.search(p, cleaned.lower()) for p in INTRO_PATTERNS)
        if is_intro:
            continue

        steps.append(cleaned)

    return steps if steps else [answer]

def get_guided_response(question: str, context_docs: list) -> dict:

    if context_docs and isinstance(context_docs[0], dict):
        context_parts = []
        for doc in context_docs:
            context_parts.append(
                f"[Page {doc['page']}, Section: {doc['section']}]\n{doc['text']}"
            )
        context = "\n\n".join(context_parts)
    else:
        context = "\n\n".join(context_docs)

    prompt = f"""You are a manufacturing SOP assistant.
Use ONLY the information in the context below to answer. Do not add any information not present in the context.
Do not say you cannot answer. Do not write any introduction or preamble.
Go straight to the numbered steps. Each step must be a concrete action.
Format your response as numbered steps only. Each step on a new line.
Start each line with the step number followed by a period.
Include exact values, settings, locations, and tool names from the context.

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