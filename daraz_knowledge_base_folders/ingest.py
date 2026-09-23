import os
import json
import uuid
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------- Config ----------------
ROOT_DIR = "daraz_knowledge_base"       # change if using mounted Drive path
OUTPUT_DIR = "faiss_index"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, fast, good default
CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 50      # overlap between consecutive chunks

DEPARTMENTS = ["returns", "delivery", "refunds", "seller", "payments", "customer_support"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(pdf_path)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_parts.append(page_text)
    return "\n".join(text_parts)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping chunks by character count, on word boundaries."""
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        # try to break on a space instead of mid-word
        if end < text_len:
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end - overlap > start else end

    return chunks


def collect_chunks(root_dir: str):
    """Walk department subfolders, extract + chunk text, attach metadata."""
    all_chunks = []
    metadata = []

    for dept in DEPARTMENTS:
        dept_path = os.path.join(root_dir, dept)
        if not os.path.isdir(dept_path):
            print(f"Warning: department folder not found, skipping: {dept_path}")
            continue

        pdf_files = [f for f in os.listdir(dept_path) if f.lower().endswith(".pdf")]

        for pdf_file in tqdm(pdf_files, desc=f"Processing {dept}"):
            pdf_path = os.path.join(dept_path, pdf_file)
            try:
                text = extract_text_from_pdf(pdf_path)
            except Exception as e:
                print(f"Failed to read {pdf_path}: {e}")
                continue

            chunks = chunk_text(text)

            for chunk in chunks:
                chunk_id = str(uuid.uuid4())
                all_chunks.append(chunk)
                metadata.append({
                    "id": chunk_id,
                    "department": dept,
                    "source_file": pdf_file,
                    "text": chunk
                })

    return all_chunks, metadata


def build_faiss_index(chunks, embed_model_name: str):
    """Embed chunks and build a FAISS index."""
    model = SentenceTransformer(embed_model_name)
    embeddings = model.encode(
        chunks,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # so we can use inner product as cosine similarity
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized inner product
    index.add(embeddings)

    return index, dim


def main():
    print("Collecting and chunking PDFs...")
    chunks, metadata = collect_chunks(ROOT_DIR)
    print(f"Total chunks created: {len(chunks)}")

    if not chunks:
        print("No chunks found. Check ROOT_DIR and folder structure.")
        return

    print("Building embeddings and FAISS index...")
    index, dim = build_faiss_index(chunks, EMBED_MODEL_NAME)

    # Save FAISS index
    index_path = os.path.join(OUTPUT_DIR, "index.faiss")
    faiss.write_index(index, index_path)

    # Save metadata (aligned by position with the FAISS index)
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Save config for later reference (e.g. which embedding model to use at query time)
    config_path = os.path.join(OUTPUT_DIR, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            "embed_model": EMBED_MODEL_NAME,
            "dim": dim,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "num_chunks": len(chunks)
        }, f, indent=2)

    print(f"Done. Index saved to {index_path}")
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
