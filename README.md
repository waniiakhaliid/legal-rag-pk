# Pakistani Legal RAG System

A retrieval-augmented generation (RAG) system for querying Pakistani legal documents in plain English or Urdu. Built to make legal information accessible to ordinary citizens — not just lawyers.

## What it does

Ask questions like:
- "What are my rights if I am arrested without a warrant?"
- "How do I file an FIR and what happens after?"
- "What is the procedure for a civil property dispute?"
- "What does the Constitution say about freedom of speech?"

The system retrieves the most relevant sections from the legal corpus and generates a grounded, cited answer using Gemini 2.0 Flash.

## Corpus

| Document | Version | Source |
|---|---|---|
| Constitution of Pakistan | 1973 (with amendments) | [na.gov.pk](https://na.gov.pk/en/constitution.php) |
| Code of Criminal Procedure | 1898 | [pakistancode.gov.pk](https://pakistancode.gov.pk) |
| Pakistan Penal Code | 1860 | [pakistancode.gov.pk](https://pakistancode.gov.pk) |
| Code of Civil Procedure | 1908 | [pakistancode.gov.pk](https://pakistancode.gov.pk) |

> **Note:** Corpus last updated: August 2026. Documents are sourced from official Pakistani government repositories. The system is designed to be re-ingested when source documents are updated — check pakistancode.gov.pk for the latest versions.

## Architecture

```
Query (English or Urdu)
        ↓
  Hybrid Retrieval
  ├── BM25 (keyword search)        — catches exact legal terms, section numbers
  └── Dense semantic search        — catches meaning-based matches
        ↓
    Reranker                       — re-scores retrieved chunks by true relevance
        ↓
  Gemini 2.0 Flash                 — generates answer grounded in retrieved chunks
        ↓
  Answer + Source Citations
```

## Technical Stack

- **Parsing:** pdfplumber
- **Chunking:** LangChain recursive text splitter (structure-aware)
- **Embeddings:** `paraphrase-multilingual-mpnet-base-v2` (multilingual, runs locally)
- **Vector DB:** ChromaDB (local)
- **Keyword search:** BM25 via rank_bm25
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM:** Gemini 2.0 Flash
- **UI:** Streamlit
- **Evaluation:** RAGAS

## Project Structure

```
legal-rag-pk/
├── data/
│   └── raw/              # source PDFs
├── src/
│   ├── ingest.py         # parse → chunk → embed → store
│   ├── retriever.py      # hybrid search + reranking
│   ├── generator.py      # prompt engineering + Gemini call
│   └── evaluate.py       # RAGAS evaluation pipeline
├── app.py                # Streamlit UI
├── .env                  # API keys (not committed)
├── requirements.txt
└── README.md
```

## Setup

**1. Clone and install**
```bash
git clone https://github.com/yourusername/legal-rag-pk
cd legal-rag-pk
uv init .
uv add pdfplumber langchain langchain-community sentence-transformers chromadb rank_bm25 google-generativeai streamlit python-dotenv ragas
```

**2. Add API key**

Create a `.env` file:
```
GEMINI_API_KEY=your_key_here
```

**3. Add documents**

Place the four PDFs in `data/raw/`.

**4. Ingest**
```bash
uv run python src/ingest.py
```

**5. Run**
```bash
uv run streamlit run app.py
```

## Evaluation

A golden Q&A set of 30 questions with known correct sources is used to measure retrieval precision via RAGAS. Run:

```bash
uv run python src/evaluate.py
```

## Limitations

- Corpus is static — does not auto-update when legislation changes
- Reranker is English-optimized; Urdu query quality depends on embedding model
- Not a substitute for actual legal advice

## Why this exists

Most Pakistanis have no practical access to their own legal rights. This system is a step toward making that information queryable in plain language, without needing a lawyer or knowing which document to look in.