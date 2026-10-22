"""
Count the chunks that ingest.py builds for each PDF (from the cached extracted
text, so no re-extraction and no Gemini calls), and compare with Pinecone.

Run from the project root:
    uv run python src/check_chunks.py            # counts only
    uv run python src/check_chunks.py --verify   # also check IDs exist in Pinecone
"""
import sys
from pathlib import Path

from pinecone import Pinecone

from config import DATA_DIR, PINECONE_API_KEY, PINECONE_INDEX_NAME
from ingest import (
    chunk_footnotes,
    chunk_text,
    extract_text_from_pdf,
    make_chunk_id,
)


def build_chunks(pdf_path: Path) -> list[dict]:
    source = pdf_path.stem
    text, footnotes = extract_text_from_pdf(pdf_path)  # uses the cache
    chunks = chunk_text(text, source_name=source)
    chunks.extend(chunk_footnotes(footnotes, source_name=source))
    return chunks


def main():
    verify = "--verify" in sys.argv

    total_chunks = 0
    all_ids = {}  # source -> set of chunk IDs

    print(f"{'PDF':40} {'section':>8} {'footnote':>9} {'total':>7} {'unique IDs':>11}")
    print("-" * 80)

    for pdf_path in sorted(Path(DATA_DIR).glob("*.pdf")):
        chunks = build_chunks(pdf_path)
        n_section = sum(c["content_type"] == "section" for c in chunks)
        n_footnote = sum(c["content_type"] == "footnote" for c in chunks)
        ids = {make_chunk_id(c["text"], c["source"]) for c in chunks}

        all_ids[pdf_path.stem] = ids
        total_chunks += len(chunks)
        print(f"{pdf_path.stem:40} {n_section:>8} {n_footnote:>9} {len(chunks):>7} {len(ids):>11}")

    unique_total = sum(len(s) for s in all_ids.values())
    print("-" * 80)
    print(f"{'ALL':40} {'':>8} {'':>9} {total_chunks:>7} {unique_total:>11}")

    index = Pinecone(api_key=PINECONE_API_KEY).Index(PINECONE_INDEX_NAME)
    pinecone_count = index.describe_index_stats()["total_vector_count"]
    print(f"\nChunks built locally:        {total_chunks}")
    print(f"Unique IDs (what Pinecone keeps): {unique_total}")
    print(f"Vectors in Pinecone:         {pinecone_count}")

    if pinecone_count == unique_total:
        print("MATCH: Pinecone holds every unique chunk.")
    elif pinecone_count < unique_total:
        print(f"MISSING: Pinecone has {unique_total - pinecone_count} fewer vectors than expected.")
    else:
        print(f"EXTRA: Pinecone has {pinecone_count - unique_total} more vectors than expected "
              "(old or stale vectors from earlier runs?).")

    if not verify:
        print("\nTip: run with --verify to see exactly which chunks are missing.")
        return

    print("\nVerifying IDs in Pinecone...")
    for source, ids in all_ids.items():
        ids = list(ids)
        found = set()
        for i in range(0, len(ids), 100):
            batch = ids[i:i + 100]
            found.update(index.fetch(ids=batch).vectors.keys())
        missing = len(ids) - len(found)
        status = "OK" if missing == 0 else f"{missing} MISSING"
        print(f"{source:40} {len(found)}/{len(ids)}  {status}")


if __name__ == "__main__":
    main()