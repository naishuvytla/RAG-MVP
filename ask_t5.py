import json, numpy as np, faiss, textwrap
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

chunks = json.load(open("chunks_meta.json", encoding="utf-8"))
index  = faiss.read_index("chunks.faiss")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
mdl = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

def embed_query(q):
    import numpy as np, faiss
    v = np.array(embed_model.encode([q], normalize_embeddings=True)[0], dtype="float32")
    faiss.normalize_L2(v.reshape(1,-1))
    return v

def retrieve(q, k=12):
    v = embed_query(q)
    sims, idxs = index.search(v.reshape(1,-1), k)
    return [(chunks[int(i)], float(s)) for i,s in zip(idxs[0], sims[0])]

def build_input(q):
    ctx = retrieve(q, 20)
    # simple doc-balanced context
    from collections import defaultdict
    by = defaultdict(list)
    for c,s in ctx: by[c["doc"]].append((c,s))
    parts = []
    for doc, items in by.items():
        items.sort(key=lambda x:x[1], reverse=True)
        for c,_ in items[:2]:
            parts.append(f"[{c['doc']}:{c['chunk_id']}] {c['text']}")
    context = "\n\n".join(parts)[:6000]
    return f"Answer the question based ONLY on the sources. Cite with the bracketed ids.\n\nQuestion: {q}\n\nSources:\n{context}\n\nAnswer:"

def generate(q):
    inp = build_input(q)
    ids = tok(inp, return_tensors="pt", truncation=True, max_length=2048).input_ids
    out = mdl.generate(ids, max_new_tokens=300, temperature=0.2, do_sample=False)
    return tok.decode(out[0], skip_special_tokens=True)

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Summarize each paper's main contributions."
    print(generate(q))
