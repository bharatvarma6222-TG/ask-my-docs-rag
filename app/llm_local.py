import ollama

MODEL = "phi3"


def generate_answer(question: str, context: str) -> str:
    prompt = f"""
Answer the question using ONLY the context.

Question:
{question}

Context:
{context}

If the context does not contain the answer, say you don't know.
"""

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]
