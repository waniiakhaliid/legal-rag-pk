# Mizaan (legal-rag-pk)

**A bilingual (English/Urdu) AI assistant for Pakistani law, built on Retrieval-Augmented Generation (RAG).**

Ask a legal question in plain English or Urdu and Mizaan retrieves the relevant provisions from the Pakistan Penal Code, Code of Criminal Procedure, Code of Civil Procedure and the Constitution of Pakistan. It then explains them in plain language, quoting the statute word for word and citing the exact sections. Answers are generated only from retrieved text, never from the model's memory.

> **Status:** Working prototype. The full pipeline (PDF ingestion → vector search → streaming answers with citations → React chat UI) runs end to end. Retrieval quality is under active improvement; see [Known Limitations](#known-limitations) and [Roadmap](#roadmap).

<!-- Add a screenshot or demo GIF here, e.g. ![Mizaan screenshot](docs/screenshot.png) -->
https://drive.google.com/file/d/1w3lYKqAHgeI-QJ8ZwkoEsC0otOoJppNk/view?usp=drive_link

---

## Highlights

- **Grounded, citable answers.** Each answer quotes the applicable sections verbatim, explains them in plain language, and then relates them cautiously to the user's situation. If no provision matches, the assistant says so instead of guessing.
- **Bilingual.** Questions can be asked in English, Urdu script, or Roman Urdu. Urdu answers keep the quoted law in its original wording while the explanation is written in Urdu.
- **Structure-aware ingestion.** PDFs are parsed section by section. Each chunk carries a header and metadata (document, section number, section title), and footnotes/amendment notes are separated into their own chunk type.
- **Smart query handling.** An LLM step rewrites everyday descriptions ("someone threatened me") into legal concepts, resolves follow-up questions using chat history, detects the language, and routes to the right statute.
- **Exact section lookup plus semantic search.** Queries like "Section 302" or "دفعہ ۳۰۲" (including Urdu/Arabic-Indic digits) trigger a metadata-filtered lookup that runs in parallel with the dense vector search.
- **Streaming UI.** The backend streams over Server-Sent Events: sources arrive first (rendered as citation chips), then the answer tokens stream in with Markdown rendering.
- **Resilient by design.** Resumable ingestion with checkpoints and idempotent vector IDs, retries on transient API errors, and automatic fallback to a second Gemini model on quota or timeout failures.

---

## How It Works

```
 INGESTION (offline)
 ───────────────────
 Legal PDFs ──► pdfplumber extraction ──► section-aware chunking ──► Gemini embeddings ──► Pinecone
 (PPC, CrPC,    (strips headers/page      (regex section split,      (gemini-embedding-001,  (cosine, 3072-d,
  CPC,           footers; separates        recursive splitter for     checkpointed every      metadata: doc type,
  Constitution)  footnotes)                long sections)             50 chunks)              section no./title)

 QUERY (online)
 ──────────────
 User question
      │
      ▼
 1. prepare_query   Gemini Flash-Lite: detect language, rewrite as a standalone English
      │             legal search query, pick the statute (or none)
      ▼
 2. retrieve        Pinecone dense search (sections only)  ┐ run in parallel
      │             Exact section-number lookup (if cited) ┘ → merge, de-duplicate, top 8
      ▼
 3. stream_answer   Gemini, strictly grounded in the numbered passages
      │             (quote → plain meaning → how it may apply), in the user's language
      ▼
 FastAPI /chat (SSE: sources → tokens → done)  ──►  React chat UI
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language / packaging | Python, [uv](https://github.com/astral-sh/uv) |
| API | FastAPI, Server-Sent Events streaming, Pydantic validation |
| Vector database | Pinecone (serverless, cosine similarity) |
| Embeddings | Google Gemini `gemini-embedding-001` (3072 dimensions) |
| LLMs | Gemini Flash (answer generation), Gemini Flash-Lite (query rewriting), plus a fallback Flash model |
| PDF parsing / chunking | pdfplumber, LangChain `RecursiveCharacterTextSplitter` |
| Front end | React + Vite, `react-markdown`, custom CSS (Fraunces / Inter / Noto Nastaliq Urdu) |

---

## Data

The corpus consists of four Pakistani legal texts, placed as PDFs in `data/raw/`:

| Document | Type |
|---|---|
| Pakistan Penal Code | `penal_code` |
| Code of Criminal Procedure | `criminal_procedure` |
| Code of Civil Procedure | `civil_procedure` |
| Constitution of Pakistan | `constitution` |

The Constitution uses a dedicated extraction mode (column cropping and header removal) because of its page layout. Extracted text is cached in `data/processed/extracted_text/` so re-runs skip PDF parsing.

---

## Getting Started

### Prerequisites

- Python 3.10+ (see `.python-version`) and [uv](https://github.com/astral-sh/uv)
- Node.js 18+ (for the front end)
- A [Google AI Studio](https://aistudio.google.com/) API key (Gemini)
- A [Pinecone](https://www.pinecone.io/) API key

### 1. Install

```bash
git clone https://github.com/waniiakhaliid/legal-rag-pk.git
cd legal-rag-pk
uv sync                      # installs Python dependencies from pyproject.toml / uv.lock
cd frontend && npm install && cd ..
```

### 2. Configure

Copy `src/.env.example` to `src/.env` and fill in your keys:

```env
GEMINI_API_KEY=your_gemini_key
PINECONE_API_KEY=your_pinecone_key
```

Model names, chunk size, index name and retrieval settings live in `src/config.py`.

### 3. Build the vector index

Put the legal PDFs in `data/raw/`, then run from the repository root:

```bash
uv run python src/ingest.py
```

Ingestion creates the Pinecone index if needed and embeds chunks with automatic checkpoints. If the API quota runs out mid-run, re-run the same command and it resumes where it stopped. Finished documents are skipped.

### 4. Run the backend

```bash
uv run uvicorn api:app --app-dir src --reload --port 8000
```

Health check: `GET http://localhost:8000/health`

### 5. Run the front end

```bash
cd frontend
npm run dev
```

Open the URL Vite prints (default `http://localhost:5173`). The front end calls the API at `http://localhost:8000/chat`; change `API_URL` in `frontend/src/LegalChat.jsx` if needed.

---

## API

`POST /chat`

```json
{
  "message": "What is the punishment for rash driving?",
  "history": [{ "role": "user", "content": "..." }, { "role": "assistant", "content": "..." }]
}
```

Returns a Server-Sent Events stream:

| Event | Payload |
|---|---|
| `sources` | List of cited passages (document, section number/title, snippet, similarity score) |
| `token` | A chunk of the streamed answer text |
| `done` | `{ "language": "en" \| "ur" }` |
| `error` | A user-facing error message |

---

## Project Structure

```
legal-rag-pk/
├── src/
│   ├── ingest.py        # PDF extraction, section-aware chunking, embedding, Pinecone upsert
│   ├── rag.py           # query rewriting, hybrid retrieval, grounded streaming generation
│   ├── api.py           # FastAPI app with the SSE /chat endpoint
│   ├── config.py        # models, index and chunking settings
│   └── cli.py           # command-line entry point
├── frontend/            # React + Vite chat UI (splash screen, streaming, citations)
├── tools/               # helper scripts for inspecting chunks and models
├── data/
│   ├── raw/             # source PDFs
│   └── processed/       # cached extracted text
├── checkpoints/         # ingestion progress markers
└── pyproject.toml
```

---

## Known Limitations

- **Chunking is the main weakness.** Sections are detected with regular expressions and long sections are split by a character-based splitter (1,000 characters, 100 overlap). Provisos, explanations, illustrations and sub-sections can end up separated from their parent clause, and some sections are mis-detected. This affects retrieval quality and is the top improvement priority.
- Retrieval returns one chunk per section, so very long provisions may only be partially visible to the model.
- The corpus covers only four core statutes; it does not include case law, provincial laws or special statutes.
- No formal evaluation set yet, so answer quality has been checked manually rather than measured.
- Embedding during ingestion is sequential (one chunk per request), which is slow for large documents.
- The front end is a minimal demo interface, and CORS is configured for localhost only.

---

## Roadmap

- [ ] Improve chunking: hierarchy-aware splitting that keeps provisos, explanations and sub-sections attached to their section, with richer metadata
- [ ] Add a reranking step (a reranker is planned; `TOP_K_RERANK` is already reserved in the config)
- [ ] Build an evaluation set of legal questions and measure retrieval hit-rate and answer faithfulness
- [ ] Batch embeddings during ingestion for faster indexing
- [ ] Expand the corpus: more federal statutes, provincial laws and court judgments
- [ ] Add Urdu-language source documents for native Urdu retrieval
- [ ] Deployment (hosted backend and front end, restricted CORS)

---

## Disclaimer

Mizaan is an educational and experimental project. It provides **general legal information only and is not legal advice**. Answers may be incomplete or inaccurate. Always consult a qualified lawyer for legal matters.

---

## Author

**Your Name** · [LinkedIn](https://linkedin.com/in/waniiakhaliid) · [GitHub](https://github.com/waniiakhaliid)

