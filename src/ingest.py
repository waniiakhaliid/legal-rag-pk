import hashlib
import re
import time
import pdfplumber
from pathlib import Path
from google import genai as google_genai
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import (
    GEMINI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_DIMENSION,
    PINECONE_CLOUD,
    PINECONE_REGION,
    GEMINI_EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DATA_DIR
)

client = google_genai.Client(api_key=GEMINI_API_KEY)

FOOTER_PATTERN = re.compile(r"^Page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE)


def strip_footer_lines(text: str) -> str:
    if not text:
        return text
    lines = text.split("\n")
    cleaned = [line for line in lines if not FOOTER_PATTERN.match(line.strip())]
    return "\n".join(cleaned)


DOC_CONFIG = {
    "PakistanPenalCode": {
        "doc_type": "penal_code",
        "doc_label": "Pakistan Penal Code",
        "section_regex": re.compile(r"(?m)^(\d{1,3}[A-Z]?)\.\s+(.*)"),
        "extraction_mode": "plain",
    },
    "CodeofCriminalProcedure": {
        "doc_type": "criminal_procedure",
        "doc_label": "Code of Criminal Procedure",
        "section_regex": re.compile(r"(?m)^(\d{1,3}[A-Z]?)\.\s+(.*)"),
        "extraction_mode": "plain",
    },
    "CodeOfCivilProcedure": {
        "doc_type": "civil_procedure",
        "doc_label": "Code of Civil Procedure",
        "section_regex": re.compile(r"(?m)^(\d{1,3}[A-Z]?)\.\s+(.*)"),
        "extraction_mode": "plain",
    },
    "Constitution_of_Pakistan_Current": {
        "doc_type": "constitution",
        "doc_label": "Constitution of Pakistan",
        "section_regex": re.compile(r"(?m)^(\d{1,3}[A-Z]{0,3})\.\s+(.*)"),
        "extraction_mode": "constitution",
    },
}

DEFAULT_CONFIG = {
    "doc_type": "unknown",
    "doc_label": "Document",
    "section_regex": re.compile(r"(?m)^(\d{1,3}[A-Z]?)\.\s+(.*)"),
    "extraction_mode": "plain",
}


def get_doc_config(source_name: str) -> dict:
    return DOC_CONFIG.get(source_name, DEFAULT_CONFIG)


SEPARATOR_CHARS = {"-", "\u2013", "\u2014", "_"}


def is_separator_line(stripped_text: str) -> bool:
    return bool(stripped_text) and set(stripped_text) <= SEPARATOR_CHARS


def find_footnote_rule_y(page) -> float | None:
    candidates = []
    for obj in page.lines:
        width = abs(obj["x1"] - obj["x0"])
        height = abs(obj["bottom"] - obj["top"])
        if width > page.width * 0.3 and height < 2:
            candidates.append(obj["top"])
    for obj in page.rects:
        width = abs(obj["x1"] - obj["x0"])
        height = abs(obj["bottom"] - obj["top"])
        if width > page.width * 0.3 and height < 2:
            candidates.append(obj["top"])
    if not candidates:
        return None
    return min(candidates)


def extract_plain_page(page) -> tuple[str, str]:
    words = page.extract_words()
    if not words:
        return "", ""

    footnote_cutoff = None
    for w in words:
        if is_separator_line(w["text"].strip()):
            footnote_cutoff = w["top"]
            break

    if footnote_cutoff is None:
        footnote_cutoff = find_footnote_rule_y(page)

    body_words = [w for w in words if footnote_cutoff is None or w["top"] < footnote_cutoff]
    footnote_words = [w for w in words if footnote_cutoff is not None and w["top"] > footnote_cutoff]

    def words_to_text(word_list):
        if not word_list:
            return ""
        lines = {}
        for w in word_list:
            lines.setdefault(round(w["top"]), []).append(w)
        out = []
        for y in sorted(lines.keys()):
            line_words = sorted(lines[y], key=lambda w: w["x0"])
            out.append(" ".join(w["text"] for w in line_words))
        return "\n".join(out)

    return words_to_text(body_words), words_to_text(footnote_words)


def extract_plain(pdf_path: Path) -> tuple[str, str]:
    text = ""
    footnotes = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            body, footnote = extract_plain_page(page)
            if body:
                text += body + "\n"
            if footnote:
                footnotes += footnote + "\n"
    return text, footnotes


CONSTITUTION_BODY_X0 = 160
CONSTITUTION_BODY_X1 = 452
CONSTITUTION_HEADER_TOP_CUTOFF = 95


def extract_constitution_page(page) -> tuple[str, str]:
    words = page.extract_words()
    if not words:
        return "", ""

    footnote_cutoff = None
    for w in words:
        if is_separator_line(w["text"].strip()):
            footnote_cutoff = w["top"]
            break

    if footnote_cutoff is None:
        footnote_cutoff = find_footnote_rule_y(page)

    body_words = [
        w for w in words
        if CONSTITUTION_BODY_X0 <= w["x0"] and w["x1"] <= CONSTITUTION_BODY_X1
        and w["top"] > CONSTITUTION_HEADER_TOP_CUTOFF
        and (footnote_cutoff is None or w["top"] < footnote_cutoff)
    ]

    footnote_words = [
        w for w in words
        if footnote_cutoff is not None and w["top"] > footnote_cutoff
    ]

    def words_to_text(word_list):
        if not word_list:
            return ""
        lines = {}
        for w in word_list:
            lines.setdefault(round(w["top"]), []).append(w)
        out = []
        for y in sorted(lines.keys()):
            line_words = sorted(lines[y], key=lambda w: w["x0"])
            out.append(" ".join(w["text"] for w in line_words))
        return "\n".join(out)

    return words_to_text(body_words), words_to_text(footnote_words)


def extract_constitution(pdf_path: Path) -> tuple[str, str]:
    text = ""
    footnotes = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            body, footnote = extract_constitution_page(page)
            if body:
                text += body + "\n"
            if footnote:
                footnotes += footnote + "\n"
    return text, footnotes


def extract_text_from_pdf(pdf_path: Path, cache_dir: Path = None, force: bool = False) -> tuple[str, str]:
    cache_dir = cache_dir or pdf_path.parent / "extracted_text"
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / f"{pdf_path.stem}.txt"
    footnote_cache_path = cache_dir / f"{pdf_path.stem}.footnotes.txt"

    if cache_path.exists() and not force:
        print(f"Loading cached text for {pdf_path.name}")
        body = cache_path.read_text(encoding="utf-8")
        footnotes = footnote_cache_path.read_text(encoding="utf-8") if footnote_cache_path.exists() else ""
        return body, footnotes

    config = get_doc_config(pdf_path.stem)
    start = time.time()

    if config["extraction_mode"] == "constitution":
        text, footnotes = extract_constitution(pdf_path)
    else:
        text, footnotes = extract_plain(pdf_path)

    text = strip_footer_lines(text)
    footnotes = strip_footer_lines(footnotes)

    elapsed = time.time() - start
    cache_path.write_text(text, encoding="utf-8")
    footnote_cache_path.write_text(footnotes, encoding="utf-8")
    print(f"Extracted: {pdf_path.name} — {len(text)} chars body, {len(footnotes)} chars footnotes, in {elapsed:.1f}s")
    return text, footnotes


def split_into_sections(text: str, section_regex: re.Pattern) -> list[dict]:
    matches = list(section_regex.finditer(text))
    sections = []

    if not matches:
        return [{"number": None, "title": None, "text": text}]

    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if len(preamble) > 50:
            sections.append({"number": None, "title": None, "text": preamble})

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        number = m.group(1).strip()
        title = m.group(2).strip()[:120]
        body = text[start:end].strip()
        sections.append({"number": number, "title": title, "text": body})

    return sections


def chunk_text(text: str, source_name: str) -> list[dict]:
    config = get_doc_config(source_name)
    sections = split_into_sections(text, config["section_regex"])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = []
    for sec in sections:
        header = f"[{config['doc_label']}"
        if sec["number"]:
            header += f" | Section {sec['number']}"
            if sec["title"]:
                header += f" - {sec['title']}"
        header += "]"

        if len(sec["text"]) <= CHUNK_SIZE:
            pieces = [sec["text"]]
        else:
            pieces = splitter.split_text(sec["text"])

        for piece in pieces:
            chunk_body = f"{header}\n{piece}".strip()
            chunks.append({
                "text": chunk_body,
                "source": source_name,
                "doc_type": config["doc_type"],
                "section_number": sec["number"] or "",
                "section_title": sec["title"] or "",
                "content_type": "section",
            })

    return chunks


def chunk_footnotes(footnote_text: str, source_name: str) -> list[dict]:
    if not footnote_text or not footnote_text.strip():
        return []

    config = get_doc_config(source_name)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    pieces = splitter.split_text(footnote_text)

    chunks = []
    header = f"[{config['doc_label']} | Amendment/Footnote Reference]"
    for piece in pieces:
        chunk_body = f"{header}\n{piece}".strip()
        chunks.append({
            "text": chunk_body,
            "source": source_name,
            "doc_type": config["doc_type"],
            "section_number": "",
            "section_title": "",
            "content_type": "footnote",
        })
    return chunks


def make_chunk_id(text: str, source: str) -> str:
    hash_input = f"{source}::{text}".encode("utf-8")
    return hashlib.md5(hash_input).hexdigest()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def get_checkpoint(source_name: str) -> int:
    checkpoint_path = Path(f"checkpoints/{source_name}.txt")
    if checkpoint_path.exists():
        return int(checkpoint_path.read_text(encoding="utf-8").strip())
    return 0


def save_checkpoint(source_name: str, chunk_index: int):
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    (checkpoint_dir / f"{source_name}.txt").write_text(str(chunk_index), encoding="utf-8")


def delete_checkpoint(source_name: str):
    checkpoint_path = Path(f"checkpoints/{source_name}.txt")
    if checkpoint_path.exists():
        checkpoint_path.unlink()


def is_done(source_name: str) -> bool:
    return Path(f"checkpoints/{source_name}.done").exists()


def mark_done(source_name: str):
    Path("checkpoints").mkdir(exist_ok=True)
    Path(f"checkpoints/{source_name}.done").write_text("done", encoding="utf-8")


# ---------------------------------------------------------------------------
# Embedding — stores to Pinecone every 50 chunks so progress is never lost
# ---------------------------------------------------------------------------

def embed_chunks(chunks: list[dict], source_name: str, index) -> list[dict]:
    start_index = get_checkpoint(source_name)
    if start_index > 0:
        print(f"Resuming from chunk {start_index} (checkpoint found)")

    for i, chunk in enumerate(chunks):
        if i < start_index:
            continue

        success = False
        for attempt in range(4):
            try:
                result = client.models.embed_content(
                    model=GEMINI_EMBEDDING_MODEL,
                    contents=chunk["text"],
                )
                chunk["embedding"] = result.embeddings[0].values
                success = True
                break
            except Exception as e:
                wait = 20 * (attempt + 1)  # 20s, 40s, 60s, 80s
                print(f"Error on chunk {i}, attempt {attempt+1}: {e} — retrying in {wait}s")
                time.sleep(wait)

        if not success:
            # store whatever we have before giving up
            embedded_so_far = [c for c in chunks[:i] if "embedding" in c]
            if embedded_so_far:
                store_in_pinecone(embedded_so_far, index)
            save_checkpoint(source_name, i)
            raise Exception(f"Failed at chunk {i} — progress saved. Resume tomorrow.")

        time.sleep(0.5)

        # every 50 chunks — store to Pinecone and save checkpoint
        if i % 50 == 0 and i > 0:
            embedded_so_far = [c for c in chunks[:i+1] if "embedding" in c]
            store_in_pinecone(embedded_so_far, index)
            save_checkpoint(source_name, i)
            print(f"Embedded and stored {i}/{len(chunks)} chunks...")

    # store any remaining chunks after the loop
    remaining = [c for c in chunks if "embedding" in c]
    if remaining:
        store_in_pinecone(remaining, index)

    delete_checkpoint(source_name)
    mark_done(source_name)
    return chunks


# ---------------------------------------------------------------------------
# Pinecone
# ---------------------------------------------------------------------------

def store_in_pinecone(chunks: list[dict], index):
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectors = [
            {
                "id": make_chunk_id(chunk["text"], chunk["source"]),
                "values": chunk["embedding"],
                "metadata": {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "doc_type": chunk["doc_type"],
                    "section_number": chunk["section_number"],
                    "section_title": chunk["section_title"],
                    "content_type": chunk.get("content_type", "section"),
                }
            }
            for chunk in batch
        ]
        index.upsert(vectors=vectors)
        print(f"Stored batch {i//batch_size + 1}...")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
        )
        print(f"Created index: {PINECONE_INDEX_NAME}")

    index = pc.Index(PINECONE_INDEX_NAME)

    for pdf_path in Path(DATA_DIR).glob("*.pdf"):
        source_name = pdf_path.stem

        if is_done(source_name):
            print(f"Skipping {pdf_path.name} — already ingested")
            continue

        print(f"\nProcessing: {pdf_path.name}")
        text, footnote_text = extract_text_from_pdf(pdf_path)

        chunks = chunk_text(text, source_name=source_name)
        footnote_chunks = chunk_footnotes(footnote_text, source_name=source_name)
        chunks.extend(footnote_chunks)
        print(f"Total chunks for {source_name}: {len(chunks)}")

        print("Embedding chunks...")
        embed_chunks(chunks, source_name, index)
        print(f"Done: {pdf_path.name}")


if __name__ == "__main__":
    main()