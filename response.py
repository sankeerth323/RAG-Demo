import faiss
import pickle
from sentence_transformers import SentenceTransformer
from groq import Groq

# ----------------------------
# Load vector database (ONCE)
# ----------------------------
_index = faiss.read_index("faq_index.faiss")

with open("faq_texts.pkl", "rb") as f:
    _texts = pickle.load(f)

with open("faq_metadata.pkl", "rb") as f:
    _metadata = pickle.load(f)

_embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# Groq client (cloud LLM)
# ----------------------------
_groq_client = Groq()

# ----------------------------
# System prompt
# ----------------------------
SYSTEM_PROMPT = """
You are a business FAQ assistant.

Rules:
- Answer ONLY using the provided context.
- If the context does not contain the answer, say:
  "I don't have enough information to answer that."
- Do NOT use outside knowledge.
- Do NOT guess.
- You may lightly rephrase for clarity, but do not change the meaning.
"""

# ----------------------------
# Retrieval
# ----------------------------
def _retrieve_context(query, k=3):
    q_emb = _embed_model.encode([query]).astype("float32")
    distances, indices = _index.search(q_emb, k)

    contexts = []
    for i in indices[0]:
        contexts.append(_texts[i])

    return contexts

# ----------------------------
# Context builder
# ----------------------------
def _build_context(query):
    contexts = _retrieve_context(query)

    if not contexts:
        return None

    return "\n\n---\n\n".join(contexts)

# ----------------------------
# Groq LLM call
# ----------------------------
def _call_groq_llm(system_prompt, user_prompt):
    completion = _groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    return completion.choices[0].message.content

# ----------------------------
# PUBLIC API (frontend calls only this)
# ----------------------------
def answer_question(query: str) -> str:
    context = _build_context(query)

    if context is None:
        return "I don't have enough information to answer that."

    user_prompt = f"""
Context:
{context}

Question:
{query}
"""

    return _call_groq_llm(SYSTEM_PROMPT, user_prompt)
