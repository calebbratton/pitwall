"""Download FIA Sporting Regulations, chunk them by article, and build the local hybrid index.

Usage: python scripts/ingest_regulations.py
Idempotent: PDFs already in data/regs/ are not re-downloaded; the index is rebuilt from scratch.
Also writes data/regs/chunks.jsonl so chunking can be inspected by eye.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path

import httpx
from pypdf import PdfReader

from src.rag.chunking import chunk_regulations
from src.rag.index import RegulationIndex
from src.rag.sources import SOURCES

REGS_DIR = Path("data/regs")

# pypdf warns loudly about harmless xref quirks in the FIA PDFs.
logging.getLogger("pypdf").setLevel(logging.ERROR)


def main() -> None:
    REGS_DIR.mkdir(parents=True, exist_ok=True)
    chunks = []
    for source in SOURCES:
        pdf = REGS_DIR / source.filename
        if not pdf.exists():
            print(f"Downloading {source.filename}")
            resp = httpx.get(source.url, follow_redirects=True, timeout=60)
            resp.raise_for_status()
            pdf.write_bytes(resp.content)
        text = "\n".join(page.extract_text() for page in PdfReader(pdf).pages)
        source_chunks = chunk_regulations(text, source)
        print(f"{source.filename}: {len(source_chunks)} chunks")
        chunks.extend(source_chunks)

    with (REGS_DIR / "chunks.jsonl").open("w") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c)) + "\n")

    print(f"Embedding and indexing {len(chunks)} chunks (first run downloads ~100MB of models)")
    index = RegulationIndex()
    index.rebuild(chunks)
    index.close()
    print("Done.")


if __name__ == "__main__":
    main()
