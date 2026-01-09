import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load verses
with open("verses.json", "r", encoding="utf-8") as f:
    verses = json.load(f)

# Load FAISS index
index = faiss.read_index("embeddings/faiss.index")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def ask_question(question, top_k=1):
    # Encode question
    query_embedding = model.encode([question])
    query_embedding = np.array(query_embedding).astype("float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(query_embedding)

    # Search
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        verse = verses[idx]
        results.append({
            "chapter": verse["chapter"],
            "verse": verse["verse"],
            "text": verse["text"],
            "purport": verse["purport"],
            "score": float(score)  # higher = better
        })

    return results

# ---- CLI LOOP ----
print("\n🕉️ Bhagavad Gita Semantic Search")
print("Type your question (or 'exit'):\n")

while True:
    question = input("❓ Question: ")
    if question.lower() == "exit":
        break

    answers = ask_question(question, top_k=3)

    print("\n📖 Best Matching Verses:\n")
    for ans in answers:
        print(f"Chapter {ans['chapter']}, Verse {ans['verse']}")
        print("-" * 50)
        print(ans["text"])
        if ans["purport"]:
            print("\n📝 Purport:\n", ans["purport"])
        print(f"\n🔢 Similarity Score: {ans['score']}")
        print("\n" + "=" * 70 + "\n")
