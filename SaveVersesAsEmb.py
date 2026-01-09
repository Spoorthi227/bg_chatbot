import requests
import json
import time
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer

BASE_URL = "https://vedicscriptures.github.io/slok"
TARGET_AUTHOR = "prabhu"

all_verses = []

print("Fetching Bhagavad Gita verses...")

for chapter in range(1, 19):
    verse = 1
    while True:
        url = f"{BASE_URL}/{chapter}/{verse}"
        response = requests.get(url)

        if response.status_code != 200:
            break  # No more verses in this chapter

        data = response.json()

        # Extract prabhu.et and prabhu.ec safely
        prabhu = data.get(TARGET_AUTHOR)
        if prabhu and "et" in prabhu:
            text = prabhu.get("et", "").strip()
            purport = prabhu.get("ec", "").strip()  # English commentary

            # You can combine for embeddings
            embedding_text = text
            if purport:
                embedding_text += "\n\nPurport:\n" + purport

            all_verses.append({
    "faiss_id": len(all_verses),  # 🔒 CRITICAL
    "_id": data.get("_id"),
    "chapter": chapter,
    "verse": verse,
    "text": text,
    "purport": purport,
    "embedding_text": embedding_text
})


            print(f"✓ Ch {chapter} V {verse}")

        verse += 1
        time.sleep(0.05)  # be polite to API

print(f"\nTotal verses collected: {len(all_verses)}")

# Save verses as JSON
with open("verses.json", "w", encoding="utf-8") as f:
    json.dump(all_verses, f, ensure_ascii=False, indent=2)

print("Saved verses to verses.json")

# Build embeddings
print("Creating embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [v["embedding_text"] for v in all_verses]
embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)


# Ensure the folder exists
os.makedirs("embeddings", exist_ok=True)
faiss.write_index(index, "embeddings/faiss.index")

print("FAISS index saved to embeddings/faiss.index")
print("DONE ✅")
