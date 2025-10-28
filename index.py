# imports for reading json creating numpy arrays using faiss and loading sentence transformers
import json, numpy as np, faiss
from sentence_transformers import SentenceTransformer

def main():
    # load the chunk metadata and texts created by ingest
    chunks = json.load(open("corpus.json", encoding="utf-8"))
    texts  = [c["text"] for c in chunks]

    # load a small fast local embedding model
    print("Loading local embedding model (all-MiniLM-L6-v2)…")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # encode all chunk texts into vectors and keep them as float32 to save memory
    # normalize embeddings means each vector already has unit length
    vecs  = np.array(model.encode(texts, normalize_embeddings=True), dtype="float32")

    # normalize again for safety and use an inner product index which equals cosine on unit vectors
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    # persist the index to disk and also save the chunk metadata for lookups
    faiss.write_index(index, "chunks.faiss")
    json.dump(chunks, open("chunks_meta.json","w",encoding="utf-8"))
    print("Indexed", len(chunks), "chunks into chunks.faiss")

if __name__ == "__main__":
    # build the index when this file is executed as a script
    main()
