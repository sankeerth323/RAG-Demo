import json

with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

faq_items = data["questions"]   # THIS is what you iterate over
texts = []
metadata = []

for item in faq_items:
    q = item["question"].strip()
    a = item["answer"].strip()

    if len(a) < 50:
        continue

    text = f"Question: {q}\n\nAnswer:\n{a}"

    texts.append(text)
    metadata.append({
        "source": "ecommerce_faq",
        "type": "faq",
        "question": q
    })

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, show_progress_bar=True)

embeddings = np.array(embeddings).astype("float32")

import faiss

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

print("Stored vectors:", index.ntotal)

faiss.write_index(index, "faq_index.faiss")

import pickle
with open("faq_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

with open("faq_texts.pkl", "wb") as f:
    pickle.dump(texts, f)


def search(query, k=3):
    q_emb = model.encode([query]).astype("float32")
    _, idx = index.search(q_emb, k)
    return [metadata[i]["question"] for i in idx[0]]

print(search("How do I sign up?"))
