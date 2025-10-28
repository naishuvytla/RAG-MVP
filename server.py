import os, json, hashlib, uuid, threading
from typing import List, Optional, Literal, Dict
import requests
import numpy as np
import faiss
import glob
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from lxml import etree

# -------- config --------
DOCS_DIR = "docs"
TEI_DIR  = "tei"
GROBID_URL = "http://localhost:8070/api/processFulltextDocument"

CHUNKS_META_PATH = "chunks_meta.json"
FAISS_PATH       = "chunks.faiss"

# ensure dirs exist
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(TEI_DIR, exist_ok=True)

# -------- model + index singletons --------
chunks: List[dict] = []
index: Optional[faiss.Index] = None
embed_model: Optional[SentenceTransformer] = None
INDEX_LOCK = threading.Lock()

def _load_index() -> bool:
    """load or init index and metadata"""
    global chunks, index, embed_model
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    if os.path.exists(CHUNKS_META_PATH) and os.path.exists(FAISS_PATH):
        chunks = json.load(open(CHUNKS_META_PATH, encoding="utf-8"))
        index = faiss.read_index(FAISS_PATH)
        return True
    # init empty index
    chunks = []
    index = faiss.IndexFlatIP(384)  # all-MiniLM-L6-v2 returns 384-dim vectors
    return False

_HAS_INDEX = _load_index()

def _embed_texts(texts: List[str]) -> np.ndarray:
    vecs = embed_model.encode(texts, normalize_embeddings=True)
    vecs = np.array(vecs, dtype="float32")
    faiss.normalize_L2(vecs)
    return vecs

# -------- TEI helpers --------
def tei_to_title_text(tei_xml: str):
    root = etree.fromstring(tei_xml.encode("utf-8"))
    ns = {"t": "http://www.tei-c.org/ns/1.0"}
    title = " ".join(root.xpath("//t:titleStmt/t:title//text()", namespaces=ns)) or "Untitled"
    body  = " ".join(root.xpath("//t:text//t:body//text()", namespaces=ns))
    txt   = (title + "\n" + body).replace("\n", " ")
    return title.strip(), " ".join(txt.split())

def make_chunks(text: str, chunk_chars=1200, overlap=250):
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i+chunk_chars])
        i += max(1, chunk_chars - overlap)
    return out

# -------- retrieval helpers (unchanged) --------
def _embed_query(q: str) -> np.ndarray:
    v = _embed_texts([q])[0]
    return v.reshape(1, -1)

def _retrieve(q: str, k: int = 40):
    v = _embed_query(q)
    sims, idxs = index.search(v, k)
    res = []
    for i, s in zip(idxs[0], sims[0]):
        c = chunks[int(i)]
        res.append({
            "doc": c["doc"], "title": c["title"], "chunk_id": c["chunk_id"],
            "text": c["text"], "score": float(s)
        })
    return res

def _doc_balanced(cands: List[dict], per_doc_cap: int = 2, max_ctx_chars: int = 8000):
    from collections import defaultdict
    by = defaultdict(list)
    for c in cands: by[c["doc"]].append(c)
    for d in by: by[d].sort(key=lambda x: x["score"], reverse=True)

    blocks, snippets, used = [], [], 0
    cites: Dict[str, str] = {}
    for d, items in by.items():
        title = items[0]["title"]
        for c in items[:per_doc_cap]:
            block = f"[{c['doc']}:{c['chunk_id']}] {c['text']}"
            blocks.append(block); used += len(block)
            snippets.append({
                "doc": c["doc"], "title": title, "chunk_id": c["chunk_id"],
                "preview": c["text"][:300].replace("\n"," "), "score": round(c["score"],3)
            })
            cites[d] = title
            if used >= max_ctx_chars: break
        if used >= max_ctx_chars: break

    context = "\n\n".join(blocks)
    citations = [{"doc": d, "title": t} for d,t in cites.items()]
    return context, snippets, citations

def _build_prompt(question: str, context: str) -> str:
    style = (
        "you are writing for a general audience. short sentences; define jargon.\n"
        "sections: big picture, paper summaries, citations.\n"
        "every factual sentence ends with a bracketed citation like [doc:chunk_id].\n"
        "if not supported by sources, say you do not know.\n"
    )
    return f"{style}\n\n# QUESTION\n{question}\n\n# SOURCES\n{context}\n\n# ANSWER"

# optional gemini generation
_GEMINI = None
try:
    from google import genai
    from google.genai import types as gtypes
    if os.getenv("GEMINI_API_KEY"): _GEMINI = genai.Client()
except Exception:
    _GEMINI = None

def _generate_with_gemini(prompt: str, model="gemini-2.0-flash") -> Optional[str]:
    if _GEMINI is None: return None
    resp = _GEMINI.models.generate_content(
        model=model, contents=prompt,
        config=gtypes.GenerateContentConfig(temperature=0.2)
    )
    return (resp.text or "").strip()

# -------- jobs state --------
class Job(BaseModel):
    id: str
    status: Literal["queued","running","done","error"]
    progress: int
    step: Optional[str] = None
    message: Optional[str] = None
    filename: str
    doc_id: Optional[str] = None

JOBS: Dict[str, Job] = {}

def _update_job(jid: str, **fields):
    j = JOBS.get(jid)
    if not j: return
    for k,v in fields.items(): setattr(j, k, v)

def _sha256(data: bytes) -> str:
    h = hashlib.sha256(); h.update(data); return h.hexdigest()

# -------- background processing --------
def _process_pdf_job(job_id: str, data: bytes, filename: str):
    try:
        _update_job(job_id, status="running", step="save", progress=5)
        # dedupe
        content_hash = _sha256(data)
        base = os.path.splitext(os.path.basename(filename))[0]
        doc_id = f"{base}-{content_hash[:8]}"

        # if already indexed by hash in filename, skip
        existing = any(c["doc"].startswith(base) and c["doc"].endswith(content_hash[:8]) for c in chunks)

        pdf_path = os.path.join(DOCS_DIR, f"{doc_id}.pdf")
        tei_path = os.path.join(TEI_DIR,  f"{doc_id}.tei.xml")

        if not existing:
            with open(pdf_path, "wb") as f: f.write(data)

            _update_job(job_id, step="grobid", progress=20)
            r = requests.post(GROBID_URL, files={"input": open(pdf_path, "rb")}, data={"consolidateCitations": 1}, timeout=120)
            r.raise_for_status()
            tei = r.text
            open(tei_path, "w", encoding="utf-8").write(tei)

            _update_job(job_id, step="parse", progress=40)
            title, text = tei_to_title_text(tei)

            _update_job(job_id, step="chunk", progress=55)
            raw_chunks = make_chunks(text)  # char chunks

            new_chunk_objs = []
            for j, c in enumerate(raw_chunks):
                new_chunk_objs.append({
                    "doc": doc_id, "title": title,
                    "chunk_id": f"{doc_id}_{j}",
                    "text": f"{title}\n{c}"
                })

            _update_job(job_id, step="embed", progress=70)
            vecs = _embed_texts([x["text"] for x in new_chunk_objs])

            _update_job(job_id, step="index", progress=85)
            with INDEX_LOCK:
                index.add(vecs)
                chunks.extend(new_chunk_objs)
                faiss.write_index(index, FAISS_PATH)
                json.dump(chunks, open(CHUNKS_META_PATH, "w", encoding="utf-8"))

        _update_job(job_id, status="done", progress=100, step="done", doc_id=doc_id,
                    message="already indexed" if existing else "indexed")
    except Exception as e:
        _update_job(job_id, status="error", message=str(e), progress=100, step="error")

# -------- fastapi app + routes --------
app = FastAPI(title="rag-mvp api")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# serve UI at /ui and redirect / -> /ui
if not os.path.exists("web"): os.makedirs("web", exist_ok=True)
@app.get("/")
def root(): return RedirectResponse(url="/ui/")
app.mount("/ui", StaticFiles(directory="web", html=True), name="ui")

class AskRequest(BaseModel):
    question: str
    k: int = 40
    per_doc_cap: int = 2
    max_ctx_chars: int = 8000
    generate: bool = True
    model: Optional[str] = "gemini-2.0-flash"

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "has_index": True,
        "num_chunks": len(chunks),
        "gemini": bool(_GEMINI is not None)
    }

@app.post("/api/ask")
def api_ask(req: AskRequest):
    cands = _retrieve(req.question, k=req.k)
    context, snippets, citations = _doc_balanced(cands, per_doc_cap=req.per_doc_cap, max_ctx_chars=req.max_ctx_chars)

    answer = None
    used_model = None
    if req.generate and _GEMINI is not None:
        prompt = _build_prompt(req.question, context)
        answer = _generate_with_gemini(prompt, model=req.model or "gemini-2.0-flash")
        used_model = req.model or "gemini-2.0-flash"

    return {
        "question": req.question,
        "generated": answer is not None,
        "answer": answer,
        "used_model": used_model,
        "snippets": snippets,
        "citations": citations
    }

# ---- upload + jobs ----
class UploadResponse(BaseModel):
    job_id: str

@app.post("/api/upload", response_model=UploadResponse)
async def api_upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="please upload a pdf")
    data = await file.read()
    if not data or len(data) < 1024:
        raise HTTPException(status_code=400, detail="file too small or empty")

    jid = str(uuid.uuid4())
    JOBS[jid] = Job(id=jid, status="queued", progress=0, filename=file.filename)

    t = threading.Thread(target=_process_pdf_job, args=(jid, data, file.filename), daemon=True)
    t.start()
    return UploadResponse(job_id=jid)

@app.get("/api/jobs/{job_id}", response_model=Job)
def api_job(job_id: str):
    j = JOBS.get(job_id)
    if not j: raise HTTPException(status_code=404, detail="job not found")
    return j

# -------- docs listing (read-only) --------
def _safe_load_chunks_meta():
    """load chunks_meta.json if present, else return []"""
    try:
        if os.path.exists(CHUNKS_META_PATH):
            return json.load(open(CHUNKS_META_PATH, encoding="utf-8"))
    except Exception:
        pass
    return []

def _count_chunks_by_doc():
    """returns dict: {doc_id: count} from chunks_meta.json"""
    counts = {}
    for c in _safe_load_chunks_meta():
        d = c.get("doc")
        if not d:
            continue
        counts[d] = counts.get(d, 0) + 1
    return counts

@app.get("/api/docs")
def api_docs_list():
    """
    list documents discovered in docs/ with basic status:
    - doc_id: base filename (without .pdf)
    - has_tei: whether tei/<doc_id>.tei.xml exists
    - num_chunks: how many chunks currently indexed for that doc (from chunks_meta.json)
    - modified_at: file mtime iso string
    """
    pdf_paths = sorted(glob.glob(os.path.join(DOCS_DIR, "*.pdf")))
    counts = _count_chunks_by_doc()
    out = []
    for p in pdf_paths:
        base = os.path.splitext(os.path.basename(p))[0]
        tei_path = os.path.join(TEI_DIR, f"{base}.tei.xml")
        has_tei = os.path.exists(tei_path)
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec="seconds")
        except Exception:
            mtime = None
        out.append({
            "doc_id": base,
            "filename": os.path.basename(p),
            "has_tei": has_tei,
            "num_chunks": counts.get(base, 0),
            "modified_at": mtime
        })
    # newest first
    out.sort(key=lambda x: (x["modified_at"] or "", x["doc_id"]), reverse=True)
    return {"docs": out}
