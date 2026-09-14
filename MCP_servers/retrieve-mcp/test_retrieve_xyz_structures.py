#!/usr/bin/env python3
"""
Test script: list structures in xyz-structure RAG and run retrieval queries
to see which structures are returned.

Usage (from MCP_servers/retrieve-mcp):
  conda activate dft_author
  python test_retrieve_xyz_structures.py

Requires: vector store built for package xyz-structure (build_vector_store.py --package xyz-structure).
"""

import sys
import logging
from pathlib import Path

# Reduce noise from retrieval module (e.g. embedding model not found)
logging.getLogger("mcp_server_retrieve.retrieval").setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent / "src"))
from mcp_server_retrieve.retrieval import (
    set_project_root,
    get_chroma_collection,
    get_package_configs,
    retrieve_vector,
    retrieve_bm25,
    merge_retrieval_results,
    keyword_reranker,
)

PACKAGE = "xyz-structure"
# Queries to test retrieval
TEST_QUERIES = [
    "CuI(NHC2) structure coordinates",
    "CuI NHC2 ja1c09505",
    "S1A oxygen evolving complex coordinates",
    "BNOO geometry",
    "Fe NH3 xyz",
    "ja3c06046 S2B",
]


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def list_structures_in_db(collection) -> tuple[int, list[str]]:
    """Get total chunk count and unique structure names (section_title = filename.xyz)."""
    try:
        data = collection.get(include=["metadatas"])
    except Exception as e:
        print(f"[WARNING] Could not fetch collection: {e}")
        return 0, []
    metadatas = data.get("metadatas") or []
    ids = data.get("ids") or []
    total_chunks = len(ids)
    names = []
    for m in metadatas:
        if isinstance(m, dict):
            title = m.get("section_title") or m.get("source_file") or ""
            if title:
                names.append(title)
    unique = sorted(set(names))
    return total_chunks, unique


def run_retrieval_and_show(query: str, chroma_path: str, top_k: int = 5) -> list[dict]:
    """Run hybrid retrieval (vector + BM25), fallback to BM25 only if embedding fails."""
    try:
        vector_results = retrieve_vector(query, k=20, package=PACKAGE, chroma_path=chroma_path)
    except FileNotFoundError as e:
        if "embedding model" in str(e).lower():
            print(f"      (Vector search skipped: embedding model not found)")
            vector_results = []
        else:
            raise
    except Exception as e:
        print(f"      (Vector search error: {e})")
        vector_results = []

    bm25_results = retrieve_bm25(query, k=20, package=PACKAGE, chroma_path=chroma_path)

    if not vector_results and not bm25_results:
        return []

    merged = merge_retrieval_results(
        vector_results or [{"id": "", "score": 0, "text": "", "metadata": {}}][:0],
        bm25_results,
        vector_weight=0.7,
    )
    reranked = keyword_reranker(merged, query, reranker_weight=0.3, html_boost=0.15)
    return reranked[:top_k]


def main():
    project_root = get_project_root()
    set_project_root(project_root)
    chroma_path = str(project_root / "authority_library" / "rag_store" / "rag_chroma")

    configs = get_package_configs()
    if PACKAGE not in configs:
        print(f"[ERROR] Package {PACKAGE!r} not in configs. Available: {list(configs.keys())}")
        return 1

    print("=" * 70)
    print("XYZ-STRUCTURE RAG TEST")
    print("=" * 70)
    print(f"Package:      {PACKAGE}")
    print(f"Chroma path:  {chroma_path}")
    print()

    # ----- 1. List all structures in the database -----
    print("--- 1. Structures in the database (from ChromaDB metadata) ---")
    try:
        client = __import__("chromadb", fromlist=["PersistentClient"]).PersistentClient(path=chroma_path)
        coll_name = configs[PACKAGE]["collection_name"]
        collection = client.get_or_create_collection(name=coll_name, metadata={"hnsw:space": "cosine"})
    except Exception as e:
        print(f"[ERROR] Could not open ChromaDB: {e}")
        return 1

    total_chunks, unique_files = list_structures_in_db(collection)
    if not unique_files:
        print("  (No documents in collection. Build the vector store first.)")
        return 0

    print(f"  Total chunks in DB: {total_chunks}")
    print(f"  Unique structure files: {len(unique_files)}\n")
    for i, name in enumerate(unique_files[:50], 1):
        print(f"    {i:3}. {name}")
    if len(unique_files) > 50:
        print(f"    ... and {len(unique_files) - 50} more")
    print()

    # ----- 2. Run retrieval for each test query -----
    print("--- 2. Retrieval results (top 5 structures per query) ---")
    for query in TEST_QUERIES:
        print(f"\n  Query: {query!r}")
        results = run_retrieval_and_show(query, chroma_path, top_k=5)
        if not results:
            print("    (No results)")
            continue
        for rank, r in enumerate(results, 1):
            meta = r.get("metadata") or {}
            title = meta.get("section_title") or meta.get("source_file") or "(no name)"
            score = r.get("score")
            score_str = f" score={score:.3f}" if score is not None else ""
            print(f"    {rank}. {title}{score_str}")
    print()

    print("=" * 70)
    print("Done.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
