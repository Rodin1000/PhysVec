"""Core retrieval functions extracted from rag_demo.py for MCP server."""

import os
import platform
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Module-level logger
logger = logging.getLogger(__name__)

# Auto-configure huggingface_hub symlinks warning based on OS
if platform.system() == "Windows":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
else:
    pass

# Configure huggingface_hub download timeout
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

import chromadb
from chromadb import Client
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ========= Paths & Constants =========
# Unified ChromaDB storage path (shared client for all packages)
# Located under authority_library/rag_store/rag_chroma
CHROMA_PATH = "authority_library/rag_store/rag_chroma"

# Project root directory (will be set at runtime)
_project_root: Optional[Path] = None


def find_project_root(start_path: Optional[Path] = None) -> Path:
    """
    Find the project root directory by looking for authority_library directory.
    
    Args:
        start_path: Starting path for search. If None, uses current working directory.
    
    Returns:
        Path to project root directory
    
    Raises:
        FileNotFoundError: If project root cannot be found
    """
    if start_path is None:
        start_path = Path.cwd()
    else:
        start_path = Path(start_path)
    
    # Search upward from start_path for authority_library directory
    current = start_path.resolve()
    while current != current.parent:
        authority_lib = current / "authority_library"
        if authority_lib.exists() and authority_lib.is_dir():
            return current
        current = current.parent
    
    # If not found, try from the file's location
    file_path = Path(__file__).resolve()
    current = file_path.parent
    while current != current.parent:
        authority_lib = current / "authority_library"
        if authority_lib.exists() and authority_lib.is_dir():
            return current
        current = current.parent
    
    raise FileNotFoundError(
        f"Could not find project root (looking for authority_library directory). "
        f"Searched from: {start_path}"
    )


def get_project_root() -> Path:
    """
    Get the project root directory. Auto-detects if not set.
    
    Returns:
        Path to project root directory
    """
    global _project_root
    if _project_root is None:
        _project_root = find_project_root()
    return _project_root


def set_project_root(project_root: Path) -> None:
    """
    Set the project root directory explicitly.
    
    Args:
        project_root: Path to project root directory
    """
    global _project_root
    project_root = Path(project_root).resolve()
    authority_lib = project_root / "authority_library"
    if not authority_lib.exists() or not authority_lib.is_dir():
        raise ValueError(
            f"Invalid project root: {project_root}. "
            f"authority_library directory not found."
        )
    _project_root = project_root


def get_package_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get package configurations with resolved paths relative to project root.
    
    This configuration is used by:
    - build_vector_store() in vector_store_builder.py (for building vector stores)
    - retrieve_hybrid() in server.py (for retrieval operations)
    
    To modify which file types are processed when building vector stores,
    edit the 'file_extensions' list for each package below.
    
    Returns:
        Dictionary of package configurations with resolved doc_root paths
    """
    project_root = get_project_root()
    return {
        "itensormps": {
            "doc_root": project_root / "authority_library" / "itensormps",
            "collection_name": "itensor_mps_docs",
            "file_extensions": [".html", ".jl", ".py", ".md", ".txt"],  # ← Edit this to change file types
        },
        "itensors": {
            "doc_root": project_root / "authority_library" / "itensors",
            "collection_name": "itensors_docs",
            "file_extensions": [".html", ".jl", ".py", ".md", ".txt"],  # ← Edit this to change file types
        },
        "netket": {
            "doc_root": project_root / "authority_library" / "NetKet",
            "collection_name": "netket_docs",
            "file_extensions": [".html", ".jl", ".py", ".md", ".txt"],  # ← Edit this to change file types
        },
        "qiskit": {
            "doc_root": project_root / "authority_library" / "qiskit",
            "collection_name": "qiskit_docs",
            "file_extensions": [".html", ".py", ".md", ".rst", ".txt"],  # ← Edit this to change file types
        },
        "qiskit-nature": {
            "doc_root": project_root / "authority_library" / "qiskit-nature",
            "collection_name": "qiskit_nature_docs",
            "file_extensions": [".html", ".py", ".md", ".rst", ".txt"],  # ← Edit this to change file types
        },
        "qiskit-algorithms": {
            "doc_root": project_root / "authority_library" / "qiskit-algorithms",
            "collection_name": "qiskit_algorithms_docs",
            "file_extensions": [".html", ".py", ".md", ".rst", ".txt"],  # ← Edit this to change file types
        },
        "qiskit-aer": {
            "doc_root": project_root / "authority_library" / "qiskit-aer",
            "collection_name": "qiskit_aer_docs",
            "file_extensions": [".html", ".py", ".md", ".rst", ".txt"],  # ← Edit this to change file types
        },
        "orca-manual": {
            "doc_root": project_root / "authority_library" / "orca-manual",
            "collection_name": "orca-manual",
            "file_extensions": [".html", ".py", ".md", ".rst", ".txt"],  # ← Edit this to change file types
        },
        "vasp-manual": {
            "doc_root": project_root / "authority_library" / "vasp-manual",
            "collection_name": "vasp-manual",
            "file_extensions": [".html", ".py", ".md", ".rst", ".txt"],  # ← Edit this to change file types
        },
        "qe-manual": {
            "doc_root": project_root / "authority_library" / "qe-manual",
            "collection_name": "qe-manual",
            "file_extensions": [".html", ".py", ".md", ".rst", ".txt"],  # ← Edit this to change file types
        },
        "yamboo-manual": {
            "doc_root": project_root / "authority_library" / "yamboo-manual",
            "collection_name": "yamboo-manual",
            "file_extensions": [".html", ".py", ".md", ".rst", ".txt"],  # ← Edit this to change file types
        },
        "xyz-structure": {
            "doc_root": project_root / "authority_library" / "xyz-structure",
            "collection_name": "xyz-structure",
            "file_extensions": [".xyz"],
        },
    }


# Package configs: define doc paths, collection names, and file types for each package
# Note: doc_root paths will be resolved relative to project root at runtime
PACKAGE_CONFIGS_BASE = {
    "itensormps": {
        "doc_root": "authority_library/itensormps",
        "collection_name": "itensor_mps_docs",
        "file_extensions": [".html", ".jl", ".py", ".md", ".txt"],
    },
    "itensors": {
        "doc_root": "authority_library/itensors",
        "collection_name": "itensors_docs",
        "file_extensions": [".html", ".jl", ".py", ".md", ".txt"],
    },
    "netket": {
        "doc_root": "authority_library/NetKet",
        "collection_name": "netket_docs",
        "file_extensions": [".html", ".py", ".md"],
    },
    "qiskit": {
        "doc_root": "authority_library/qiskit",
        "collection_name": "qiskit_docs",
        "file_extensions": [".html", ".py", ".md", ".rst", ".txt"],
    },
    "qiskit-nature": {
        "doc_root": "authority_library/qiskit-nature",
        "collection_name": "qiskit_nature_docs",
        "file_extensions": [".html", ".py", ".md", ".rst", ".txt"],
    },
    "qiskit-algorithms": {
        "doc_root": "authority_library/qiskit-algorithms",
        "collection_name": "qiskit_algorithms_docs",
        "file_extensions": [".html", ".py", ".md", ".rst", ".txt"],
    },
    "qiskit-aer": {
        "doc_root": "authority_library/qiskit-aer",
        "collection_name": "qiskit_aer_docs",
        "file_extensions": [".html", ".py", ".md", ".rst", ".txt"],
    },
    "orca-manual": {
        "doc_root": "authority_library/orca-manual",
        "collection_name": "orca-manual",
        "file_extensions": [".html", ".py", ".md", ".rst", ".txt"],
    },
    "xyz-structure": {
        "doc_root": "authority_library/xyz-structure",
        "collection_name": "xyz-structure",
        "file_extensions": [".xyz"],
    },
}

# Default package
DEFAULT_PACKAGE = "itensormps"

# ========= Vector Embedding Model =========
# Offline‑only embedding model loading.
# We never attempt to contact Hugging Face here.

EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
_embed_model: Optional[SentenceTransformer] = None
_model_loaded: bool = False


def _get_local_model_path() -> Path:
    """Return the local embedding model path for offline use.

    Shared with vector_store_builder: the model is kept at

        <project_root>/embedding_model

    where <project_root> is the directory containing "authority_library".
    """
    current = Path(__file__).resolve()
    for _ in range(10):
        if (current / "authority_library").is_dir():
            project_root = current
            break
        if current.parent == current:
            break
        current = current.parent
    else:
        project_root = current

    return project_root / "embedding_model"


def get_embed_model() -> SentenceTransformer:
    """Lazy load the sentence‑transformers model from disk (offline).

    - Uses the local directory returned by ``_get_local_model_path``.
    - Raises a clear error if the model is missing.
    - Does **not** try to download from Hugging Face under any circumstances.
    """
    global _embed_model, _model_loaded
    if _embed_model is None:
        import logging
        logger = logging.getLogger(__name__)

        local_model_path = _get_local_model_path()
        config_file = local_model_path / "config.json"

        if not local_model_path.exists() or not config_file.exists():
            logger.error(
                "Local embedding model not found. Expected files at: %s", local_model_path
            )
            raise FileNotFoundError(
                "Local embedding model not found. Expected files at: "
                f"{local_model_path}. Make sure config.json and model.safetensors "
                "are present in that directory."
            )

        logger.info("Loading embedding model from local directory (offline): %s", local_model_path)
        _embed_model = SentenceTransformer(str(local_model_path))
        _model_loaded = True
        logger.info("Embedding model loaded successfully from local directory.")

    return _embed_model


def preload_embed_model() -> None:
    """
    Preload embedding model at program startup.
    This avoids loading delay on first query.
    
    Note: Model loading may take a few seconds even if already downloaded,
    as it needs to load into memory. This is normal and expected.
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Preloading embedding model (this may take a few seconds if first time, or ~2-5 seconds if cached)...")
    try:
        _ = get_embed_model()
        logger.info("Embedding model preloaded successfully")
    except Exception as e:
        logger.error(f"Failed to preload embedding model: {e}")
        logger.error("This may be due to network issues, missing dependencies, or insufficient memory.")
        raise


def embed_query(text: str) -> List[float]:
    """
    Embed query text.
    BGE officially recommends adding 'query: ' prefix to query.
    """
    model = get_embed_model()
    q = f"query: {text}"
    emb = model.encode([q], normalize_embeddings=True)[0]
    return emb.tolist()


# ========= ChromaDB Management =========
_chroma_client: Optional[Client] = None

# BM25 Index Management (per package)
_bm25_indices: Dict[str, Any] = {}
_bm25_chunk_maps: Dict[str, List[Dict[str, Any]]] = {}


def tokenize_text(text: str) -> List[str]:
    """Tokenize text for BM25 (simple word tokenization)."""
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


def get_chroma_client(chroma_path: Optional[str] = None) -> Client:
    """Get global ChromaDB client (singleton pattern)."""
    global _chroma_client
    if _chroma_client is None:
        path = chroma_path if chroma_path else CHROMA_PATH
        _chroma_client = chromadb.PersistentClient(path=path)
    return _chroma_client


def get_chroma_collection(package: Optional[str] = None, chroma_path: Optional[str] = None) -> chromadb.Collection:
    """
    Get ChromaDB collection for specified package.
    
    Args:
        package: Package name (e.g. "itensormps", "itensors", "netket")
                If None, use default package
        chroma_path: Optional ChromaDB storage path
    
    Returns:
        ChromaDB Collection object
    """
    if package is None:
        package = DEFAULT_PACKAGE
    
    configs = get_package_configs()
    if package not in configs:
        raise ValueError(
            f"Unknown package name: {package}. "
            f"Available packages: {list(configs.keys())}"
        )
    
    client = get_chroma_client(chroma_path)
    collection_name = configs[package]["collection_name"]
    
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def get_bm25_index(package: str, chroma_path: Optional[str] = None) -> Optional[BM25Okapi]:
    """Get BM25 index for specified package. Auto-rebuilds from ChromaDB if not in memory."""
    global _bm25_indices, _bm25_chunk_maps
    
    # If index exists in memory, return it
    if package in _bm25_indices:
        return _bm25_indices[package]
    
    # Try to rebuild from ChromaDB
    logger.debug(f"BM25 index not in memory, rebuilding from ChromaDB for package '{package}'...")
    try:
        rebuild_bm25_from_chromadb(package, chroma_path)
        return _bm25_indices.get(package)
    except Exception as e:
        logger.warning(f"Failed to rebuild BM25 index from ChromaDB: {e}")
        return None


def build_bm25_index(package: str, chunks: List[Any], chroma_path: Optional[str] = None) -> None:
    """
    Build BM25 index for a package from chunks.
    
    Args:
        package: Package name
        chunks: List of Chunk objects (from dataclass or dict-like with 'text' attribute)
        chroma_path: Optional ChromaDB storage path
    """
    global _bm25_indices, _bm25_chunk_maps
    
    if not chunks:
        return
    
    # Extract text from chunks (handle both Chunk objects and dicts)
    chunk_texts = []
    chunk_data = []
    for c in chunks:
        if hasattr(c, 'text'):
            text = c.text
            chunk_data.append({
                'id': getattr(c, 'id', ''),
                'text': text,
                'source_file': getattr(c, 'source_file', ''),
                'section_title': getattr(c, 'section_title', ''),
                'section_id': getattr(c, 'section_id', ''),
            })
        elif isinstance(c, dict):
            text = c.get('text', '')
            chunk_data.append({
                'id': c.get('id', ''),
                'text': text,
                'source_file': c.get('source_file', ''),
                'section_title': c.get('section_title', ''),
                'section_id': c.get('section_id', ''),
            })
        else:
            continue
        chunk_texts.append(text)
    
    # Tokenize all chunk texts
    tokenized_corpus = [tokenize_text(text) for text in chunk_texts]
    
    # Build BM25 index
    bm25 = BM25Okapi(tokenized_corpus)
    _bm25_indices[package] = bm25
    
    # Store chunk mapping for retrieval
    _bm25_chunk_maps[package] = [
        {
            'id': c['id'],
            'text': c['text'],
            'metadata': {
                'source_file': c['source_file'],
                'section_title': c['section_title'],
                'section_id': c['section_id'],
                'package': package,
            }
        }
        for c in chunk_data
    ]
    
    logger.debug(f"BM25 index built for package '{package}' with {len(chunks)} chunks.")


def rebuild_bm25_from_chromadb(package: str, chroma_path: Optional[str] = None) -> None:
    """
    Rebuild BM25 index from existing ChromaDB collection.
    
    Args:
        package: Package name
        chroma_path: Optional ChromaDB storage path
    """
    global _bm25_indices, _bm25_chunk_maps
    
    collection = get_chroma_collection(package, chroma_path)
    
    try:
        all_data = collection.get()
        
        if not all_data or not all_data.get('ids'):
            logger.warning(f"No documents found in ChromaDB for package '{package}'")
            return
        
        ids = all_data['ids']
        documents = all_data.get('documents', [])
        metadatas = all_data.get('metadatas', [])
        
        if not documents:
            logger.warning(f"No document texts found in ChromaDB for package '{package}'")
            return
        
        logger.debug(f"Rebuilding BM25 index from {len(documents)} documents in ChromaDB...")
        
        # Build BM25 index from documents
        tokenized_corpus = [tokenize_text(doc) for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)
        _bm25_indices[package] = bm25
        
        # Store chunk mapping for retrieval
        _bm25_chunk_maps[package] = [
            {
                'id': doc_id,
                'text': doc_text,
                'metadata': meta if i < len(metadatas) else {},
            }
            for i, (doc_id, doc_text, meta) in enumerate(zip(ids, documents, metadatas))
        ]
        
        logger.debug(f"BM25 index rebuilt successfully from ChromaDB!")
        
    except Exception as e:
        logger.error(f"Failed to rebuild BM25 index from ChromaDB: {e}")
        raise


# ========= Retrieval Functions =========
def retrieve_vector(query: str, k: int = 8, package: Optional[str] = None, chroma_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieve relevant documents using vector similarity search (embedding-based).
    
    Args:
        query: Query text
        k: Number of results to return
        package: Package name, use default if None
        chroma_path: Optional ChromaDB storage path
    
    Returns:
        List of retrieval results, each containing id, score, text, metadata
        Score is cosine distance (lower is better)
    """
    collection = get_chroma_collection(package, chroma_path)
    
    # Check if collection is empty before querying
    try:
        collection_count = collection.count()
        if collection_count == 0:
            logger.warning(f"ChromaDB collection for package '{package}' is empty. No vector retrieval results.")
            return []
    except Exception as e:
        logger.warning(f"Failed to check collection count for package '{package}': {e}")
        # Continue anyway, let the query attempt to handle it
    
    query_vector = embed_query(query)

    try:
        result = collection.query(
            query_embeddings=[query_vector],
            n_results=k,
        )
    except Exception as e:
        error_msg = str(e)
        if "Nothing found on disk" in error_msg or "hnsw" in error_msg.lower():
            logger.error(f"ChromaDB collection for package '{package}' appears to be corrupted or empty. Error: {error_msg}")
            logger.info(f"Please rebuild the vector store for package '{package}' using build_vector_store.py")
            return []
        else:
            # Re-raise other errors
            raise

    docs = result["documents"][0] if result.get("documents") and len(result["documents"]) > 0 else []
    metadatas = result["metadatas"][0] if result.get("metadatas") and len(result["metadatas"]) > 0 else []
    ids = result["ids"][0] if result.get("ids") and len(result["ids"]) > 0 else []
    distances = result.get("distances", [[None] * len(docs)])[0] if result.get("distances") and len(result["distances"]) > 0 else [None] * len(docs)

    items = []
    for i, (doc, meta) in enumerate(zip(docs, metadatas)):
        items.append(
            {
                "id": ids[i] if i < len(ids) else f"unknown_{i}",
                "score": distances[i] if i < len(distances) else None,
                "text": doc,
                "metadata": meta,
            }
        )
    return items


def retrieve_bm25(query: str, k: int = 8, package: Optional[str] = None, chroma_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieve relevant documents using BM25 keyword matching.
    
    Args:
        query: Query text
        k: Number of results to return
        package: Package name, use default if None
        chroma_path: Optional ChromaDB storage path
    
    Returns:
        List of retrieval results, each containing id, score, text, metadata
        Score is BM25 score (higher is better)
    """
    if package is None:
        package = DEFAULT_PACKAGE
    
    configs = get_package_configs()
    if package not in configs:
        raise ValueError(
            f"Unknown package name: {package}. "
            f"Available packages: {list(configs.keys())}"
        )
    
    bm25 = get_bm25_index(package, chroma_path)
    if bm25 is None:
        logger.warning(f"BM25 index not available for package '{package}'. Please build vector store first.")
        return []
    
    chunk_map = _bm25_chunk_maps.get(package, [])
    if not chunk_map:
        return []
    
    # Tokenize query
    query_tokens = tokenize_text(query)
    
    if not query_tokens:
        return []
    
    # Get BM25 scores
    scores = bm25.get_scores(query_tokens)
    
    # Create results with scores
    results = []
    for i, score in enumerate(scores):
        if score > 0:
            results.append({
                'id': chunk_map[i]['id'],
                'score': score,
                'text': chunk_map[i]['text'],
                'metadata': chunk_map[i]['metadata'],
            })
    
    # Sort by score (descending, higher is better for BM25)
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Return top k
    return results[:k]


def merge_retrieval_results(
    vector_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    vector_weight: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Merge results from vector similarity and BM25 keyword retrieval channels.
    
    Args:
        vector_results: Results from vector search (score is cosine distance, lower is better)
        bm25_results: Results from BM25 search (score is BM25 score, higher is better)
        vector_weight: Weight for vector channel (0.0-1.0)
    
    Returns:
        Merged and deduplicated results list with combined scores
    """
    # Normalize vector scores (invert distance to similarity, then normalize to [0, 1])
    # Store normalized scores for storage (all scores will be in [0, 1] range, similarity form, higher is better)
    if vector_results:
        vector_scores = [r['score'] for r in vector_results if r['score'] is not None]
        if vector_scores:
            max_vector = max(vector_scores)
            min_vector = min(vector_scores)
            vector_range = max_vector - min_vector if max_vector > min_vector else 1.0
            for r in vector_results:
                if r['score'] is not None:
                    # Normalize to similarity (higher is better) in [0, 1] range
                    r['normalized_vector_score'] = (max_vector - r['score']) / vector_range
                    # Also store normalized vector_score (similarity form, [0, 1])
                    r['normalized_vector_score_for_storage'] = r['normalized_vector_score']
                else:
                    r['normalized_vector_score'] = 0.0
                    r['normalized_vector_score_for_storage'] = 0.0
    
    # Normalize BM25 scores to [0, 1] range (similarity form, higher is better)
    if bm25_results:
        bm25_scores = [r['score'] for r in bm25_results if r['score'] is not None]
        if bm25_scores:
            max_bm25 = max(bm25_scores)
            min_bm25 = min(bm25_scores)
            bm25_range = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1.0
            for r in bm25_results:
                if r['score'] is not None:
                    # Normalize to [0, 1] range (similarity form, higher is better)
                    r['normalized_bm25_score'] = (r['score'] - min_bm25) / bm25_range
                    # Also store normalized bm25_score (similarity form, [0, 1])
                    r['normalized_bm25_score_for_storage'] = r['normalized_bm25_score']
                else:
                    r['normalized_bm25_score'] = 0.0
                    r['normalized_bm25_score_for_storage'] = 0.0
            bm25_norm_min = 0.0
            bm25_norm_max = 1.0
    
    # Create a map to merge results by ID
    merged_map: Dict[str, Dict[str, Any]] = {}
    
    # Add vector results
    bm25_weight = 1.0 - vector_weight
    for r in vector_results:
        chunk_id = r['id']
        vector_score = r.get('normalized_vector_score', 0.0)
        merged_map[chunk_id] = {
            **r,
            # Store normalized vector_score (similarity form, [0, 1], higher is better)
            'vector_score': r.get('normalized_vector_score_for_storage', 0.0),
            'bm25_score': None,
            'combined_score': vector_score * vector_weight,
        }
    
    # Merge BM25 results
    for r in bm25_results:
        chunk_id = r['id']
        bm25_score = r.get('normalized_bm25_score', 0.0)
        if chunk_id in merged_map:
            # Store normalized bm25_score (similarity form, [0, 1], higher is better)
            merged_map[chunk_id]['bm25_score'] = r.get('normalized_bm25_score_for_storage', 0.0)
            merged_map[chunk_id]['combined_score'] += bm25_score * bm25_weight
        else:
            merged_map[chunk_id] = {
                **r,
                'vector_score': None,
                # Store normalized bm25_score (similarity form, [0, 1], higher is better)
                'bm25_score': r.get('normalized_bm25_score_for_storage', 0.0),
                'combined_score': bm25_score * bm25_weight,
            }
    
    # Convert to list and sort by combined score (higher is better)
    merged_results = list(merged_map.values())
    merged_results.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # Store normalized scores (all in similarity form, [0, 1], higher is better)
    # combined_score is already in [0, 1] range (similarity form)
    for r in merged_results:
        # Store original_score as similarity (higher is better) in [0, 1] range
        r['original_score'] = r['combined_score']
        # Store score as similarity (higher is better) in [0, 1] range
        r['score'] = r['combined_score']
    
    return merged_results


def keyword_reranker(
    results: List[Dict[str, Any]], 
    query: str, 
    reranker_weight: float = 0.3,
    html_boost: float = 0.15
) -> List[Dict[str, Any]]:
    """
    Rerank retrieval results based on keyword matching in query and file type.
    
    Args:
        results: List of retrieval results
        query: Original query text
        reranker_weight: Weight controlling reranker influence (0.0-1.0)
        html_boost: Additional boost for HTML files (0.0-1.0)
    
    Returns:
        Reranked results list with adjusted scores
    """
    # Extract keywords from query
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'how', 'what', 'when', 'where', 'why'}
    query_lower = query.lower()
    words = re.findall(r'\b\w+\b', query_lower)
    keywords = [w for w in words if len(w) >= 3 and w not in stop_words]
    
    # Calculate keyword match scores for each result
    reranked = []
    for item in results:
        text_lower = item['text'].lower()
        title_lower = item['metadata'].get('section_title', '').lower()
        combined_text = f"{title_lower} {text_lower}"
        
        # Count keyword matches
        matched_keywords = []
        if keywords:
            for keyword in keywords:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, combined_text):
                    matched_keywords.append(keyword)
        match_count = len(matched_keywords)
        match_ratio = match_count / len(keywords) if keywords else 0
        
        # Check if source file is HTML
        source_file = item['metadata'].get('source_file', '')
        is_html = source_file.lower().endswith('.html')
        
        # Original retrieval score (similarity form, higher is better, range [0, 1])
        # item['score'] is similarity form (combined_score, range [0, 1], higher is better)
        # item['original_score'] is also similarity form (combined_score, range [0, 1], higher is better)
        original_score_similarity = item.get('score', 0.0) if item.get('score') is not None else 0.0
        # Keep original_score for reference
        original_score_reference = item.get('original_score', original_score_similarity)
        
        # Apply reranker boost
        keyword_boost = match_ratio * reranker_weight
        html_boost_value = html_boost if is_html else 0.0
        total_boost = min(keyword_boost + html_boost_value, 1.0)
        
        # Adjust similarity score: increase similarity based on boost
        # original_score_similarity is in [0, 1], higher is better
        adjusted_similarity = min(original_score_similarity + total_boost * (1.0 - original_score_similarity), 1.0)
        
        reranked.append({
            **item,
            'score': adjusted_similarity,  # Similarity form (higher is better), range [0, 1]
            'original_score': original_score_reference,  # Similarity form (higher is better), normalized to [0, 1]
            'keyword_matches': match_count,
            'matched_keywords': matched_keywords,
            'all_keywords': keywords,
            'match_ratio': match_ratio,
            'is_html': is_html,
        })
    
    # Sort by adjusted score (higher is better, similarity form)
    # Use reverse=True to sort from highest to lowest similarity
    reranked.sort(key=lambda x: x['score'] if x['score'] is not None else 0.0, reverse=True)
    
    return reranked


def merge_multi_package_results(
    all_package_results: List[List[Dict[str, Any]]],
    vector_weight: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Merge retrieval results from multiple packages.
    
    This function takes results from multiple packages (each already merged from vector and BM25),
    re-normalizes scores across all packages, and merges them into a single ranked list.
    
    Args:
        all_package_results: List of result lists, each from a single package's hybrid retrieval
                           Each result list contains items with 'score', 'vector_score', 'bm25_score', etc.
        vector_weight: Weight for vector channel (0.0-1.0), used for reference (scores are already merged)
    
    Returns:
        Merged results from all packages, sorted by score (similarity form, higher is better)
    """
    if not all_package_results:
        return []
    
    # Flatten all results from all packages
    all_results = []
    for package_results in all_package_results:
        all_results.extend(package_results)
    
    if not all_results:
        return []
    
    # Re-normalize scores across all packages to ensure fair comparison
    # Extract all scores for normalization
    all_scores = [r.get('score', 0.0) for r in all_results if r.get('score') is not None]
    
    if all_scores:
        max_score = max(all_scores)
        min_score = min(all_scores)
        score_range = max_score - min_score if max_score > min_score else 1.0
        
        # Re-normalize all scores to [0, 1] range
        for r in all_results:
            if r.get('score') is not None:
                original_score = r['score']
                # Normalize to [0, 1] range (similarity form, higher is better)
                r['score'] = (original_score - min_score) / score_range if score_range > 0 else 0.0
                # Update original_score to reflect the re-normalized value
                r['original_score'] = r['score']
            else:
                r['score'] = 0.0
                r['original_score'] = 0.0
    
    # Sort by normalized score (higher is better)
    all_results.sort(key=lambda x: x.get('score', 0.0) if x.get('score') is not None else 0.0, reverse=True)
    
    return all_results

