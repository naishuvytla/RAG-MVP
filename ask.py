import json, numpy as np, faiss
from sentence_transformers import SentenceTransformer

chunks = json.load(open("chunks_meta.json", encoding="utf-8"))
index  = faiss.read_index("chunks.faiss")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_query(q):
    v = np.array(embed_model.encode([q], normalize_embeddings=True)[0], dtype="float32")
    faiss.normalize_L2(v.reshape(1,-1))
    return v

def retrieve(q, k=6):
    v = embed_query(q)
    sims, idxs = index.search(v.reshape(1,-1), k)
    return [(chunks[int(i)], float(s)) for i,s in zip(idxs[0], sims[0])]

def answer(q, k=60, per_doc_cap=2, max_chars=1200):
    # 1) retrieve a larger candidate pool
    ctx = retrieve(q, k)

    print("\n=== QUESTION ===\n" + q)
    print("\n=== PER-PAPER HIGHLIGHTS ===")

    # 2) group by document and take best N per doc
    from collections import defaultdict
    by_doc = defaultdict(list)
    for c,score in ctx:
        by_doc[c["doc"]].append((c, score))

    seen = set()
    total = 0
    # sort each doc's items by score desc, then print top per_doc_cap
    for doc, items in by_doc.items():
        items.sort(key=lambda x: x[1], reverse=True)
        title = items[0][0]["title"] if items else doc
        print(f"\n# {title}  (source: {doc})")
        show = items[:per_doc_cap]
        if not show:
            print("- (no strong matches)")
        for c,score in show:
            snippet = c["text"][:300].replace("\n"," ")
            print(f"- {snippet}…  (cos={score:.3f})")
            seen.add(f"{title} (source: {doc})")
            total += len(snippet)
            if total >= max_chars: break
        if total >= max_chars: break

    print("\n=== CITATIONS ===")
    for s in seen: print("-", s)

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What are the main contributions and methods in this paper?"
    answer(q)
