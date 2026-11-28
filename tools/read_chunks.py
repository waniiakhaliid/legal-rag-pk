"""
Save ALL chunks of one document from Pinecone to a text file, in section order.

Run from the project root:
    uv run python src/read_chunks.py
"""
import re
import sys

from pinecone import Pinecone

from config import PINECONE_API_KEY, PINECONE_INDEX_NAME

# ---------------------------------------------------------------------------
# SETTINGS: edit these
# ---------------------------------------------------------------------------
DOC_TYPE = "penal_code"   # "penal_code" | "criminal_procedure" | "civil_procedure"
                          # | "constitution" | None (all documents)
CONTENT_TYPE = None       # "section" | "footnote" | None (both)
OUTPUT_FILE = "text.txt"
# ---------------------------------------------------------------------------

sys.stdout.reconfigure(encoding="utf-8")
index = Pinecone(api_key=PINECONE_API_KEY).Index(PINECONE_INDEX_NAME)

stats = index.describe_index_stats()
print("dimension:", stats.dimension, "| total vectors:", stats.total_vector_count)
print("namespaces:", {k: v.vector_count for k, v in stats.namespaces.items()})

NAMESPACES = list(stats.namespaces.keys()) or [""]
for extra in ("", "__default__"):
    if extra not in NAMESPACES:
        NAMESPACES.append(extra)


def matches(md: dict) -> bool:
    if DOC_TYPE and md.get("doc_type") != DOC_TYPE:
        return False
    if CONTENT_TYPE and md.get("content_type") != CONTENT_TYPE:
        return False
    return True


def fetch_all_via_list(ns: str) -> list[tuple[str, dict]]:
    rows, scanned = [], 0
    try:
        for batch in index.list(namespace=ns):
            ids = list(batch)
            for i in range(0, len(ids), 100):
                res = index.fetch(ids=ids[i:i + 100], namespace=ns)
                for vid, vec in res.vectors.items():
                    scanned += 1
                    md = vec.metadata or {}
                    if matches(md):
                        rows.append((vid, md))
            print(f"  scanned {scanned} vectors, kept {len(rows)}...", end="\r")
    except Exception as e:
        print(f"\n[list failed for namespace {ns!r}] {type(e).__name__}: {str(e)[:150]}")
    print()
    return rows if scanned else []


def fetch_all_via_query(ns: str) -> list[tuple[str, dict]]:
    """Fallback if list() is unavailable. Pinecone caps one query at 1000 results."""
    flt = {}
    if DOC_TYPE:
        flt["doc_type"] = {"$eq": DOC_TYPE}
    if CONTENT_TYPE:
        flt["content_type"] = {"$eq": CONTENT_TYPE}
    res = index.query(
        vector=[0.1] * stats.dimension, top_k=1000,
        include_metadata=True, filter=flt or None, namespace=ns,
    )
    if len(res.matches) >= 1000:
        print("WARNING: hit the 1000-result query limit, output may be incomplete.")
    return [(m.id, m.metadata or {}) for m in res.matches]


rows: list[tuple[str, dict]] = []
for ns in NAMESPACES:
    rows = fetch_all_via_list(ns)
    if rows:
        print(f"Read via list (namespace={ns!r})")
        break
if not rows:
    for ns in NAMESPACES:
        rows = fetch_all_via_query(ns)
        if rows:
            print(f"Read via query (namespace={ns!r})")
            break

if not rows:
    print("No chunks found. Check DOC_TYPE / CONTENT_TYPE and the stats above.")
    sys.exit(1)


# ---- sort by section order: 2 < 10 < 22 < 22A < 337 < 337O < 498B ---------
def section_key(md: dict):
    sec = str(md.get("section_number", "") or "")
    m = re.match(r"(\d+)\s*([A-Za-z]*)", sec)
    if not m:
        return (1, 0, "")                      # no section number: goes last
    return (0, int(m.group(1)), m.group(2).upper())


def sort_key(row):
    vid, md = row
    return (
        md.get("doc_type", ""),
        section_key(md),
        md.get("content_type", ""),
        md.get("chunk_index", 0) if isinstance(md.get("chunk_index", 0), (int, float)) else 0,
        md.get("page", 0) if isinstance(md.get("page", 0), (int, float)) else 0,
        md.get("text", "")[:80],
    )


rows.sort(key=sort_key)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(f"Document: {DOC_TYPE or 'ALL'} | content_type: {CONTENT_TYPE or 'ALL'} | "
            f"chunks: {len(rows)}\n\n")
    for n, (vid, md) in enumerate(rows, 1):
        f.write("=" * 90 + "\n")
        f.write(f"#{n}  ID: {vid}\n")
        f.write(f"doc_type: {md.get('doc_type', '')} | content_type: {md.get('content_type', '')} | "
                f"section: {md.get('section_number', '') or '-'} | {md.get('section_title', '')}\n")
        f.write("-" * 90 + "\n")
        f.write(md.get("text", "") + "\n\n")

print(f"Saved {len(rows)} chunks to {OUTPUT_FILE}")