# imports for environment access json vectors faiss local embedding and gemini client
import os, json, numpy as np, faiss
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

# load saved chunk metadata and the faiss index built earlier
chunks = json.load(open("chunks_meta.json", encoding="utf-8"))
index  = faiss.read_index("chunks.faiss")

# load the same local embedding model to embed user questions
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_query(q):
    # encode the question text as a single vector and normalize to unit length
    v = np.array(embed_model.encode([q], normalize_embeddings=True)[0], dtype="float32")
    faiss.normalize_L2(v.reshape(1,-1))
    return v

def retrieve(q, k=30):
    # search the index for the top k most similar chunks to the question
    v = embed_query(q)
    sims, idxs = index.search(v.reshape(1,-1), k)
    # return a list of tuples containing the chunk metadata and the similarity score
    return [(chunks[int(i)], float(s)) for i,s in zip(idxs[0], sims[0])]

def build_prompt(question, k=30, per_doc_cap=2, max_ctx_chars=8000):
    # gather candidate chunks and group them by document so each paper is represented
    from collections import defaultdict
    by_doc = defaultdict(list)
    for c,score in retrieve(question, k):
        by_doc[c["doc"]].append((c,score))
    # sort within each document group by similarity descending
    for d in by_doc:
        by_doc[d].sort(key=lambda x:x[1], reverse=True)

    # stitch together a context from the best few chunks per document while respecting a size budget
    parts, used = [], 0
    for d, items in by_doc.items():
        for c,_ in items[:per_doc_cap]:
            block = f"[{c['doc']}:{c['chunk_id']}] {c['text']}"
            parts.append(block)
            used += len(block)
            if used >= max_ctx_chars:
                break
        if used >= max_ctx_chars:
            break

    # join all blocks into one context string
    context = "\n\n".join(parts)

    # create simple instructions that ask the model to answer only from the provided sources and to cite inline
    sys = (
        "you are a careful scientific assistant. answer only using the sources below. "
        "cite sources inline with their bracketed ids for example [doc:chunk_id]. "
        "if something is not supported by the sources say you do not know."
    )

    # return the final prompt that includes instructions question and stitched sources
    return f"{sys}\n\n# QUESTION\n{question}\n\n# SOURCES\n{context}\n\n# ANSWER"

def generate_with_gemini(prompt, model="gemini-2.0-flash"):
    # create a gemini client which reads the api key from the environment
    client = genai.Client()
    # call the model with a low temperature so answers stay grounded in the sources
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2)
    )
    # return the plain text part of the response
    return resp.text

if __name__ == "__main__":
    # read the question from the command line and fall back to a default if none is given
    import sys
    q = " ".join(sys.argv[1:]) or "state each paper’s main contributions in three bullets."
    # build a prompt from retrieved sources and ask gemini to generate a concise cited answer
    prompt = build_prompt(q)
    print("\n=== QUESTION ===\n" + q)
    print("\n=== ANSWER ===\n" + generate_with_gemini(prompt))
