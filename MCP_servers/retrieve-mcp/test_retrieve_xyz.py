#!/usr/bin/env python3
"""Quick test: retrieve CuI(NHC2) structure from xyz-structure RAG (paper ja1c09505_s1)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from mcp_server_retrieve.retrieval import (
    set_project_root,
    retrieve_vector,
    retrieve_bm25,
    merge_retrieval_results,
    keyword_reranker,
)

def main():
    project_root = Path(__file__).resolve().parent.parent.parent  # PrlAuthor
    set_project_root(project_root)
    chroma_path = str(project_root / "authority_library" / "rag_store" / "rag_chroma")

    # Query for CuI(NHC2) from paper ja1c09505_s1 (note: we have ja1c09505_s1, not ja1c09006_s1)
    query = "CuI(NHC2) structure coordinates from ja1c09505_s1"
    package = "xyz-structure"
    print(f"Query: {query!r}")
    print(f"Package: {package}")
    print(f"Chroma path: {chroma_path}\n")

    vector_results = retrieve_vector(query, k=15, package=package, chroma_path=chroma_path)
    bm25_results = retrieve_bm25(query, k=15, package=package, chroma_path=chroma_path)
    merged = merge_retrieval_results(vector_results, bm25_results, vector_weight=0.7)
    reranked = keyword_reranker(merged, query, reranker_weight=0.3, html_boost=0.15)
    results = reranked[:5]

    print(f"Retrieved {len(results)} chunks (top 5):\n")
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {}) or {}
        source = meta.get("source_file", "")
        title = meta.get("section_title", "")
        text_preview = (r.get("text", "") or "")[:220].replace("\n", " ")
        print(f"  {i}. source_file={source!r} section_title={title!r}")
        print(f"     text: {text_preview}...")
        print()

    found = any(
        "ja1c09505_s1_CuI_NHC2" in (meta.get("section_title") or "") or "ja1c09505_s1_CuI_NHC2" in (meta.get("source_file") or "")
        for r in results
        for meta in [r.get("metadata") or {}]
    )
    print("Top-5 contains CuI(NHC2) / ja1c09505_s1_CuI_NHC2.xyz:", found)

if __name__ == "__main__":
    main()
