# rag-mvp (work in progress)

A tiny **RAG literature chatbot**. It parses PDFs to TEI with **Dockerized GROBID**, chunks text with overlap, embeds using **sentence-transformers**, indexes in **FAISS**, and serves **semantic search & Q&A** via **FastAPI** with a **Tailwind** UI (front-end uploads, live indexing). Optional answers use **Gemini** with inline `[doc:chunk]` citations.

## Features
- **Semantic search over PDFs** (FAISS + sentence-transformers)  
- **TEI XML parsing** (GROBID) and high-recall chunking  
- **Web UI**: upload, indexing status, ask questions  
- **Grounded answers** (optional Gemini) with citations
