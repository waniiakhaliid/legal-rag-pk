# Pakistani Legal RAG: Ingestion & Embedding

The first stage of a retrieval-augmented generation (RAG) system that will let ordinary people query Pakistani law in plain English or Urdu.

**Current status:** ingestion and embedding only. This stage reads the legal PDFs, splits them into section-level chunks, embeds them with Gemini, and stores them in Pinecone. Retrieval, answer generation and the chat UI are planned (see [Roadmap](#roadmap)).

## Corpus

| Document | Version | Source |
|---|---|---|
| Constitution of Pakistan | 1973 (with amendments) | [na.gov.pk](https://na.gov.pk/en/constitution.php) |
| Code of Criminal Procedure | 1898 | [pakistancode.gov.pk](https://pakistancode.gov.pk) |
| Pakistan Penal Code | 1860 | [pakistancode.gov.pk](https://pakistancode.gov.pk) |
| Code of Civil Procedure | 1908 | [pakistancode.gov.pk](https://pakistancode.gov.pk) |

The PDFs are not committed to this repository. Download them from the sources above.

## How ingestion works

```
PDF
 ↓  pdfplumber: extract text, separate body from footnotes
 ↓  cache in data/raw/extracted_text/
 ↓  regex: split into law sections (e.g. "302. Punishment for murder")
 ↓  LangChain splitter: break very long sections into smaller pieces
 ↓  prefix each chunk with a label: [Pakistan Penal Code | Section 302 - ...]
 ↓  Gemini embedding API: one vector per chunk
 ↓  Pinecone: upsert vectors + metadata
```

**Key details**

- **Footnotes are kept separately.** Footnotes and amendment references are cut from the body and stored as their own chunks (`content_type = "footnote"`), so amendments stay searchable without polluting the section text.
- **The Constitution has its own extractor.** Its pages use a different layout, so it crops to a fixed body area and ignores headers.
- **Chunk IDs are content hashes** (`md5(source + text)`). Re-embedding the same chunk overwrites the same record, so reruns never create duplicates.
- **Metadata stored with each vector:** `text`, `source`, `doc_type`, `section_number`, `section_title`, `content_type`.

## Resuming after an interruption

Embedding makes one Gemini call per chunk, so free-tier quotas can stop a run partway through. The script is built to resume safely. It uses three kinds of local files:

| File | Meaning |
|---|---|
| `data/raw/extracted_text/<name>.txt` | Text already extracted from the PDF |
| `checkpoints/<name>.txt` | Progress in the current PDF (chunk number) |
| `checkpoints/<name>.done` | This PDF is completely ingested; skip it |

- Progress is saved to Pinecone and the checkpoint every 50 chunks, and also when all retries fail.
- Failed calls are retried 4 times, waiting 20, 40, 60 and 80 seconds.
- On the next run, finished PDFs are skipped and unfinished ones resume from the checkpoint.

> Resuming relies on chunks being rebuilt in the same order. If you change a PDF, `CHUNK_SIZE` or `CHUNK_OVERLAP` mid-run, delete the `checkpoints/` folder and start that PDF again.

## Project structure

```
LegalRag/
├── data/
│   └── raw/                # source PDFs (not committed)
│       └── extracted_text/ # cached extraction output
├── checkpoints/            # resume markers (not committed)
├── src/
│   ├── config.py           # settings and API keys
│   └── ingest.py           # extract → chunk → embed → store
├── .env                    # API keys (not committed)
├── pyproject.toml
└── uv.lock
```

## Setup

**1. Install dependencies** (uses [uv](https://docs.astral.sh/uv/))

```bash
git clone <your-repo-url>
cd LegalRag
uv sync
```

**2. Get API keys**

- Gemini: [aistudio.google.com](https://aistudio.google.com)
- Pinecone: [pinecone.io](https://pinecone.io) (the script creates the index if it does not exist)

**3. Create a `.env` file in the project root**

```
GEMINI_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here
```

Other settings (index name, embedding model and dimension, chunk size and overlap, data folder) live in `src/config.py`.

**4. Add the PDFs** to `data/raw/`. The file names must match these, because the extraction and labeling rules are keyed by file name:

- `PakistanPenalCode.pdf`
- `CodeofCriminalProcedure.pdf`
- `CodeOfCivilProcedure.pdf`
- `Constitution_of_Pakistan_Current.pdf`

**5. Run ingestion**

```bash
uv run python src/ingest.py
```

If the Gemini quota runs out, the script stops and saves its progress. Run the same command again later to continue.

## Changing the embedding model

Vectors from different models cannot be mixed. If you change `GEMINI_EMBEDDING_MODEL` (or move to a different provider), then:

1. Create a new Pinecone index with the new `EMBEDDING_DIMENSION`.
2. Delete the `checkpoints/` folder.
3. Re-run ingestion.

## Limitations

- Embedding is one API call per chunk, so a full ingestion can take several runs on a free tier.
- Section detection uses regular expressions on the extracted text. Pages with unusual layouts may be split imperfectly, so spot-check a few chunks in Pinecone after ingesting.
- The corpus is static. When legislation changes, download the new PDFs and re-ingest.
- This is a research and learning project, not legal advice.

## Roadmap

- [ ] Retrieval: Pinecone semantic search, optional BM25, reranking
- [ ] Answer generation with section citations
- [ ] Chat UI
- [ ] Evaluation on a golden question set (English and Urdu)