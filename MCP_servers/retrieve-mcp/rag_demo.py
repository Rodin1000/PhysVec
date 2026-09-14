import os
import sys
import platform
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

# Auto-configure huggingface_hub symlinks warning based on OS
# Windows doesn't support symlinks by default, disable warning; Linux/macOS use symlinks for better performance
if platform.system() == "Windows":
    # Check if Windows supports symlinks (e.g., developer mode enabled)
    # Use user setting if env var is set; otherwise check symlink support
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
else:
    # Linux/macOS: keep default behavior, use symlinks for better cache performance
    # Don't set env var, let huggingface_hub use default symlinks behavior
    pass

# Configure huggingface_hub download timeout (avoid network timeout errors)
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")  # 120 seconds timeout

import requests
import chromadb
from chromadb import Client
import tiktoken
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
import re

# ========= Paths & Constants =========
# Unified ChromaDB storage path (shared client for all packages)
CHROMA_PATH = "rag_store/rag_chroma"

# Package configs: define doc paths, collection names, and file types for each package
PACKAGE_CONFIGS = {
    "itensormps": {
        "doc_root": Path("authority_library/itensormps"),
        "collection_name": "itensor_mps_docs",
        "file_extensions": [".html", ".jl", ".py", ".md", ".txt"],  # Supported file extensions
    },
    "itensors": {
        "doc_root": Path("authority_library/itensors"),
        "collection_name": "itensors_docs",
        "file_extensions": [".html", ".jl", ".py", ".md", ".txt"],  # Multiple file types supported
    },
    "netket": {
        "doc_root": Path("authority_library/NetKet"),
        "collection_name": "netket_docs",
        "file_extensions": [".html", ".py", ".md"],  # Supported file types
    },
    "qiskit": {
        "doc_root": Path("authority_library/qiskit"),
        "collection_name": "qiskit_docs",
        "file_extensions": [".html", ".py", ".md", ".rst", ".txt"],
    },
    "qiskit-nature": {
        "doc_root": Path("authority_library/qiskit-nature"),
        "collection_name": "qiskit_nature_docs",
        "file_extensions": [".html", ".py", ".md", ".rst", ".txt"],
    },
    "qiskit-algorithms": {
        "doc_root": Path("authority_library/qiskit-algorithms"),
        "collection_name": "qiskit_algorithms_docs",
        "file_extensions": [".html", ".py", ".md", ".rst", ".txt"],
    },
    "qiskit-aer": {
        "doc_root": Path("authority_library/qiskit-aer"),
        "collection_name": "qiskit_aer_docs",
        "file_extensions": [".html", ".py", ".md", ".rst", ".txt"],
    },
}

# Default package (for backward compatibility)
DEFAULT_PACKAGE = "itensormps"
DOC_ROOT = PACKAGE_CONFIGS[DEFAULT_PACKAGE]["doc_root"]  # Backward compatibility
COLLECTION_NAME = PACKAGE_CONFIGS[DEFAULT_PACKAGE]["collection_name"]  # Backward compatibility

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_CHAT_MODEL = "openai/gpt-4o-mini"

# ========= Vector Embedding Model (BGE for English) =========

EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
_embed_model: Optional[SentenceTransformer] = None
_model_loaded: bool = False  # Track if model has been loaded

def get_embed_model() -> SentenceTransformer:
    """
    Lazy load sentence-transformers model (singleton pattern).
    Downloads from Hugging Face on first call if not cached locally.
    Subsequent calls return the cached model without reloading.
    """
    global _embed_model, _model_loaded
    if _embed_model is None:
        if not _model_loaded:
            print(f"[DEBUG] Loading embedding model: {EMBED_MODEL_NAME} ...")
            print(f"[DEBUG] Note: Model will be cached in memory for subsequent queries.")
        try:
            _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
            _model_loaded = True
            print(f"[DEBUG] Model loaded successfully! (Cached in memory)")
        except Exception as e:
            print(f"[ERROR] Failed to load embedding model: {e}")
            raise
    # Model already loaded, using cached version (no message needed)
    return _embed_model

def preload_embed_model() -> None:
    """
    Preload embedding model at program startup.
    This avoids loading delay on first query.
    """
    _ = get_embed_model()

def embed_passages(texts: List[str]) -> List[List[float]]:
    """
    Embed document chunks (passages).
    BGE officially recommends adding 'passage: ' prefix to text.
    """
    model = get_embed_model()
    prefixed = [f"passage: {t}" for t in texts]
    emb = model.encode(prefixed, normalize_embeddings=True)
    return [e.tolist() for e in emb]

def embed_query(text: str) -> List[float]:
    """
    Embed query text.
    BGE officially recommends adding 'query: ' prefix to query.
    """
    model = get_embed_model()
    q = f"query: {text}"
    emb = model.encode([q], normalize_embeddings=True)[0]
    return emb.tolist()


# ========= tiktoken: Token counting =========

enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Estimate token count for text, used for chunk splitting."""
    return len(enc.encode(text))

def split_long_text(text: str,
                    max_tokens: int = 600,
                    overlap_tokens: int = 100) -> List[str]:
    """
    Split long text into chunks by token length with overlap.
    Used in HTML parsing.
    """
    tokens = enc.encode(text)
    chunks: List[str] = []
    n = len(tokens)
    start = 0

    while start < n:
        end = min(start + max_tokens, n)
        sub_tokens = tokens[start:end]
        chunks.append(enc.decode(sub_tokens))
        if end == n:
            break
        start = end - overlap_tokens  # Backtrack to create overlap

    return chunks

# ========= OpenRouter Chat Wrapper =========
def call_openrouter_chat(messages: List[Dict[str, str]],
                         model: str = DEFAULT_CHAT_MODEL,
                         temperature: float = 0.2,
                         max_tokens: int = 1024) -> str:
    """
    Call OpenRouter /chat/completions API, return assistant text content.
    messages structure is similar to OpenAI Chat API, e.g.:
      [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
      ]
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Please set OPENROUTER_API_KEY in environment variables")

    try:
        client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"calling model {model} failed with error: {str(e)}"


# Step 3
@dataclass
class Chunk:
    id: str            # Globally unique ID (based on file absolute path)
    text: str          # Plain text of this chunk
    source_file: str   # Source file (relative to doc root)
    section_title: str # Section title (may be function/class name for code files)
    section_id: str    # Section identifier

def generate_unique_chunk_id(file_path: Path, section_id: str, chunk_index: int) -> str:
    """
    Generate unique chunk ID based on file absolute path.
    
    Args:
        file_path: Full path of the file
        section_id: Section identifier
        chunk_index: Chunk index
    
    Returns:
        Unique chunk ID
    """
    import hashlib
    # Use absolute path to ensure uniqueness
    abs_path = str(file_path.resolve())
    # Use full MD5 hash of path to ensure uniqueness (32 hex digits)
    path_hash = hashlib.md5(abs_path.encode('utf-8')).hexdigest()
    # Combine filename and hash for uniqueness while maintaining readability
    file_name = file_path.stem  # Without extension to avoid special characters
    # Combine to generate unique ID
    return f"{file_name}__{path_hash}__{section_id}__{chunk_index}"

def extract_sections_from_html(html_path: Path, doc_root: Path) -> List[Chunk]:
    """
    Extract structured document chunks from HTML file.
    
    Args:
        html_path: Full path of HTML file
        doc_root: Document root directory for calculating relative paths
    """
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")

    # Remove navigation/sidebar (adjust selectors based on actual document structure)
    for selector in ["nav", "header", "footer", ".sidebar", ".sidenav", ".toc"]:
        for tag in soup.select(selector):
            tag.decompose()

    body = soup.body or soup

    chunks: List[Chunk] = []
    current_title = "ROOT"
    current_section_id = "root"
    buffer_lines: List[str] = []
    sec_index = 0

    def flush_buffer() -> List[Chunk]:
        nonlocal buffer_lines, sec_index, current_title, current_section_id
        if not buffer_lines:
            return []

        full_text = "\n".join(buffer_lines).strip()
        buffer_lines = []

        if not full_text:
            return []

        # Split by token if too long
        pieces: List[str]
        if count_tokens(full_text) > 800:
            pieces = split_long_text(full_text, max_tokens=600, overlap_tokens=100)
        else:
            pieces = [full_text]

        new_chunks: List[Chunk] = []
        for i, piece in enumerate(pieces):
            chunk_id = generate_unique_chunk_id(html_path, f"{current_section_id}_{sec_index}", i)
            new_chunks.append(
                Chunk(
                    id=chunk_id,
                    text=piece,
                    source_file=str(html_path.relative_to(doc_root)),
                    section_title=current_title,
                    section_id=current_section_id,
                )
            )
        sec_index += 1
        return new_chunks

    # Scan body in order
    for el in body.descendants:
        if not getattr(el, "name", None):
            continue

        if el.name == "h2":
            # New section: flush previous buffer to chunks
            chunks.extend(flush_buffer())
            current_title = el.get_text(strip=True) or "UNTITLED"
            current_section_id = el.get("id") or f"sec_{sec_index}"
            # Add h2 title text to buffer so it's included in chunks for better retrieval
            if current_title and current_title != "UNTITLED":
                buffer_lines.append(f"## {current_title}")
        elif el.name in {"p", "li"}:
            txt = el.get_text(" ", strip=True)
            if txt:
                buffer_lines.append(txt)
        elif el.name == "pre":
            code_txt = el.get_text("\n", strip=True)
            if code_txt:
                # Add code block marker for potential reranking
                buffer_lines.append("```code\n" + code_txt + "\n```")

    chunks.extend(flush_buffer())
    return chunks

def extract_sections_from_code(code_path: Path, doc_root: Path) -> List[Chunk]:
    """
    Extract document chunks from code files (.py, .jl, etc.).
    Split by function/class/module level.
    
    Args:
        code_path: Full path of code file
        doc_root: Document root directory for calculating relative paths
    """
    try:
        content = code_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[WARNING] Failed to read {code_path}: {e}")
        return []
    
    if not content.strip():
        return []
    
    chunks: List[Chunk] = []
    file_ext = code_path.suffix.lower()
    
    # If file is too long, split by line count first
    lines = content.split('\n')
    if len(lines) > 1000:
        # For very long files, split by fixed line count
        chunk_size = 500
        overlap = 50
        for i in range(0, len(lines), chunk_size - overlap):
            chunk_lines = lines[i:min(i + chunk_size, len(lines))]
            chunk_text = '\n'.join(chunk_lines)
            if chunk_text.strip():
                chunk_index = i // chunk_size
                section_id = f"lines_{i}_{i+len(chunk_lines)}"
                chunk_id = generate_unique_chunk_id(code_path, section_id, chunk_index)
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        text=chunk_text,
                        source_file=str(code_path.relative_to(doc_root)),
                        section_title=f"Lines {i+1}-{min(i+len(chunk_lines), len(lines))}",
                        section_id=section_id,
                    )
                )
    else:
        # For shorter files, treat entire file as one chunk, may need further splitting
        if count_tokens(content) > 800:
            pieces = split_long_text(content, max_tokens=600, overlap_tokens=100)
            for i, piece in enumerate(pieces):
                section_id = f"part_{i}"
                chunk_id = generate_unique_chunk_id(code_path, section_id, i)
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        text=piece,
                        source_file=str(code_path.relative_to(doc_root)),
                        section_title=code_path.name,
                        section_id=section_id,
                    )
                )
        else:
            chunk_id = generate_unique_chunk_id(code_path, "full", 0)
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=content,
                    source_file=str(code_path.relative_to(doc_root)),
                    section_title=code_path.name,
                    section_id="full",
                )
            )
    
    return chunks

def extract_sections_from_markdown(md_path: Path, doc_root: Path) -> List[Chunk]:
    """
    Extract document chunks from Markdown file.
    Split by headings (#, ##, ###).
    
    Args:
        md_path: Full path of Markdown file
        doc_root: Document root directory for calculating relative paths
    """
    try:
        content = md_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[WARNING] Failed to read {md_path}: {e}")
        return []
    
    if not content.strip():
        return []
    
    chunks: List[Chunk] = []
    lines = content.split('\n')
    current_title = md_path.stem  # Use filename as default title
    current_section_id = "root"
    buffer_lines: List[str] = []
    sec_index = 0
    
    def flush_buffer() -> List[Chunk]:
        nonlocal buffer_lines, sec_index, current_title, current_section_id
        if not buffer_lines:
            return []
        
        full_text = "\n".join(buffer_lines).strip()
        buffer_lines = []
        
        if not full_text:
            return []
        
        # Split by token if too long
        pieces: List[str]
        if count_tokens(full_text) > 800:
            pieces = split_long_text(full_text, max_tokens=600, overlap_tokens=100)
        else:
            pieces = [full_text]
        
        new_chunks: List[Chunk] = []
        for i, piece in enumerate(pieces):
            chunk_id = generate_unique_chunk_id(md_path, f"{current_section_id}_{sec_index}", i)
            new_chunks.append(
                Chunk(
                    id=chunk_id,
                    text=piece,
                    source_file=str(md_path.relative_to(doc_root)),
                    section_title=current_title,
                    section_id=current_section_id,
                )
            )
        sec_index += 1
        return new_chunks
    
    for line in lines:
        stripped = line.strip()
        # Detect Markdown headings
        if stripped.startswith('#'):
            chunks.extend(flush_buffer())
            # Extract heading level and text
            level = len(line) - len(line.lstrip('#'))
            title_text = stripped.lstrip('#').strip()
            if title_text:
                current_title = title_text
                current_section_id = f"h{level}_{sec_index}"
            else:
                current_title = "UNTITLED"
                current_section_id = f"h{level}_{sec_index}"
        else:
            if stripped:  # Ignore empty lines
                buffer_lines.append(line)
    
    chunks.extend(flush_buffer())
    return chunks

def extract_sections_from_text(txt_path: Path, doc_root: Path) -> List[Chunk]:
    """
    Extract document chunks from plain text file.
    
    Args:
        txt_path: Full path of text file
        doc_root: Document root directory for calculating relative paths
    """
    try:
        content = txt_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[WARNING] Failed to read {txt_path}: {e}")
        return []
    
    if not content.strip():
        return []
    
    # For text files, split directly by token
    if count_tokens(content) > 800:
        pieces = split_long_text(content, max_tokens=600, overlap_tokens=100)
        chunks = []
        for i, piece in enumerate(pieces):
            section_id = f"part_{i}"
            chunk_id = generate_unique_chunk_id(txt_path, section_id, i)
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=piece,
                    source_file=str(txt_path.relative_to(doc_root)),
                    section_title=txt_path.name,
                    section_id=section_id,
                )
            )
        return chunks
    else:
        chunk_id = generate_unique_chunk_id(txt_path, "full", 0)
        return [
            Chunk(
                id=chunk_id,
                text=content,
                source_file=str(txt_path.relative_to(doc_root)),
                section_title=txt_path.name,
                section_id="full",
            )
        ]

def collect_all_chunks(doc_root: Path, file_extensions: List[str] = None) -> List[Chunk]:
    """
    Collect document chunks from all specified file types in directory.
    
    Args:
        doc_root: Document root directory path
        file_extensions: List of supported file extensions, e.g. [".html", ".py", ".jl", ".md", ".txt"]
                        If None, defaults to processing only .html files
    """
    if file_extensions is None:
        file_extensions = [".html"]
    
    all_chunks: List[Chunk] = []
    all_files: List[Path] = []
    
    # Collect all matching files
    for ext in file_extensions:
        pattern = f"**/*{ext}"
        files = list(doc_root.glob(pattern))
        all_files.extend(files)
        print(f"Found {len(files)} {ext} files")
    
    # Exclude certain files
    excluded_names = {"search.html", "__pycache__", ".git", ".svn", ".hg"}
    filtered_files = [
        f for f in all_files 
        if f.name.lower() not in excluded_names 
        and not any(excluded in str(f) for excluded in excluded_names)
    ]
    
    print(f"Total files to process: {len(filtered_files)}")
    
    # Process by file type
    for path in filtered_files:
        ext = path.suffix.lower()
        try:
            if ext == ".html":
                file_chunks = extract_sections_from_html(path, doc_root)
            elif ext in [".py", ".jl", ".js", ".ts", ".cpp", ".c", ".h", ".hpp", ".java", ".go", ".rs"]:
                file_chunks = extract_sections_from_code(path, doc_root)
            elif ext in [".md", ".markdown"]:
                file_chunks = extract_sections_from_markdown(path, doc_root)
            elif ext in [".txt", ".text"]:
                file_chunks = extract_sections_from_text(path, doc_root)
            else:
                # For unknown types, try treating as text file
                print(f"[WARNING] Unknown file type {ext}, treating as text: {path}")
                file_chunks = extract_sections_from_text(path, doc_root)
            
            all_chunks.extend(file_chunks)
        except Exception as e:
            print(f"[ERROR] Failed to process {path}: {e}")
            continue

    print(f"Total chunks extracted: {len(all_chunks)}")
    return all_chunks


# Step 4: ChromaDB Management
# Global client (singleton pattern, shared by all packages)
_chroma_client: Optional[Client] = None

# BM25 Index Management (per package)
_bm25_indices: Dict[str, Any] = {}  # package -> BM25Okapi instance
_bm25_chunk_maps: Dict[str, List[Dict[str, Any]]] = {}  # package -> list of {id, text, metadata}

def tokenize_text(text: str) -> List[str]:
    """Tokenize text for BM25 (simple word tokenization)."""
    # Simple tokenization: split on whitespace and punctuation, lowercase
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

def get_bm25_index(package: str) -> Optional[BM25Okapi]:
    """Get BM25 index for specified package. Auto-rebuilds from ChromaDB if not in memory."""
    global _bm25_indices, _bm25_chunk_maps
    
    # If index exists in memory, return it
    if package in _bm25_indices:
        return _bm25_indices[package]
    
    # Try to rebuild from ChromaDB
    print(f"[DEBUG] BM25 index not in memory, rebuilding from ChromaDB for package '{package}'...")
    try:
        rebuild_bm25_from_chromadb(package)
        return _bm25_indices.get(package)
    except Exception as e:
        print(f"[WARNING] Failed to rebuild BM25 index from ChromaDB: {e}")
        return None

def rebuild_bm25_from_chromadb(package: str) -> None:
    """
    Rebuild BM25 index from existing ChromaDB collection.
    
    This function allows BM25 retrieval to work without re-running build_vector_store.
    It reads all documents from ChromaDB and rebuilds the BM25 index in memory.
    
    Args:
        package: Package name
    """
    global _bm25_indices, _bm25_chunk_maps
    
    collection = get_chroma_collection(package)
    
    # Get all documents from ChromaDB
    # Note: ChromaDB doesn't have a direct "get all" method, so we query with a large limit
    # or we can use get() without filters to get all items
    try:
        # Get all items from collection
        all_data = collection.get()
        
        if not all_data or not all_data.get('ids'):
            print(f"[WARNING] No documents found in ChromaDB for package '{package}'")
            return
        
        ids = all_data['ids']
        documents = all_data.get('documents', [])
        metadatas = all_data.get('metadatas', [])
        
        if not documents:
            print(f"[WARNING] No document texts found in ChromaDB for package '{package}'")
            return
        
        print(f"[DEBUG] Rebuilding BM25 index from {len(documents)} documents in ChromaDB...")
        
        # Create Chunk objects from ChromaDB data
        chunks = []
        for i, (doc_id, doc_text) in enumerate(zip(ids, documents)):
            meta = metadatas[i] if i < len(metadatas) else {}
            chunk = Chunk(
                id=doc_id,
                text=doc_text,
                source_file=meta.get('source_file', 'unknown'),
                section_title=meta.get('section_title', ''),
                section_id=meta.get('section_id', ''),
            )
            chunks.append(chunk)
        
        # Build BM25 index from chunks
        build_bm25_index(package, chunks)
        print(f"[DEBUG] BM25 index rebuilt successfully from ChromaDB!")
        
    except Exception as e:
        print(f"[ERROR] Failed to rebuild BM25 index from ChromaDB: {e}")
        raise

def build_bm25_index(package: str, chunks: List[Chunk]) -> None:
    """
    Build BM25 index for a package from chunks.
    
    Args:
        package: Package name
        chunks: List of Chunk objects
    """
    global _bm25_indices, _bm25_chunk_maps
    
    if not chunks:
        return
    
    # Tokenize all chunk texts
    tokenized_corpus = [tokenize_text(c.text) for c in chunks]
    
    # Build BM25 index
    bm25 = BM25Okapi(tokenized_corpus)
    _bm25_indices[package] = bm25
    
    # Store chunk mapping for retrieval
    _bm25_chunk_maps[package] = [
        {
            'id': c.id,
            'text': c.text,
            'metadata': {
                'source_file': c.source_file,
                'section_title': c.section_title,
                'section_id': c.section_id,
                'package': package,
            }
        }
        for c in chunks
    ]
    
    print(f"[DEBUG] BM25 index built for package '{package}' with {len(chunks)} chunks.")

def get_chroma_client() -> Client:
    """Get global ChromaDB client (singleton pattern)."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _chroma_client

def get_chroma_collection(package: Optional[str] = None) -> chromadb.Collection:
    """
    Get ChromaDB collection for specified package.
    
    Args:
        package: Package name (e.g. "itensormps", "itensors", "netket")
                If None, use default package (backward compatibility)
    
    Returns:
        ChromaDB Collection object
    """
    if package is None:
        package = DEFAULT_PACKAGE
    
    if package not in PACKAGE_CONFIGS:
        raise ValueError(
            f"Unknown package name: {package}. "
            f"Available packages: {list(PACKAGE_CONFIGS.keys())}"
        )
    
    client = get_chroma_client()
    collection_name = PACKAGE_CONFIGS[package]["collection_name"]
    
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # Matches our normalized embeddings
    )
    return collection

def list_all_collections() -> List[str]:
    """List all existing collection names."""
    client = get_chroma_client()
    collections = client.list_collections()
    return [col.name for col in collections]


def build_vector_store(package: Optional[str] = None, doc_root: Optional[Path] = None):
    """
    Build vector database for specified package.
    
    Args:
        package: Package name (e.g. "itensormps", "itensors", "netket")
                If None, use default package
        doc_root: Document root directory (optional, read from config if not provided)
    """
    if package is None:
        package = DEFAULT_PACKAGE
    
    if package not in PACKAGE_CONFIGS:
        raise ValueError(
            f"Unknown package name: {package}. "
            f"Available packages: {list(PACKAGE_CONFIGS.keys())}"
        )
    
    config = PACKAGE_CONFIGS[package]
    if doc_root is None:
        doc_root = config["doc_root"]
    
    # Get supported file types from config
    file_extensions = config.get("file_extensions", [".html"])
    
    print(f"[DEBUG] Step 1/4: Getting collection '{config['collection_name']}'...")
    collection = get_chroma_collection(package)
    
    print(f"[DEBUG] Step 2/4: Collecting chunks from '{doc_root}' (file types: {file_extensions})...")
    chunks = collect_all_chunks(doc_root, file_extensions=file_extensions)
    print(f"[DEBUG] Step 2.5/4: Building BM25 index for {len(chunks)} chunks...")
    build_bm25_index(package, chunks)
    print(f"[DEBUG] Step 3/4: Generating vector embeddings for {len(chunks)} chunks...")

    batch_size = 64
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c.text for c in batch]
        metas = [
            {
                "package": package,  # Add package identifier
                "source_file": c.source_file,
                "section_title": c.section_title,
                "section_id": c.section_id,
            }
            for c in batch
        ]
        ids = [c.id for c in batch]

        batch_num = i // batch_size + 1
        if batch_num % 5 == 0 or batch_num == total_batches:
            print(f"[DEBUG] Processing batch {batch_num}/{total_batches}...")
        
        embeddings = embed_passages(texts)
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metas,
        )

    print(f"[DEBUG] Step 4/4: Done! Inserted {len(chunks)} chunks.")



# Step 5: Retrieval Functions
def retrieve_vector(query: str, k: int = 8, package: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieve relevant documents using vector similarity search (embedding-based).
    
    Args:
        query: Query text
        k: Number of results to return
        package: Package name, use default if None
    
    Returns:
        List of retrieval results, each containing id, score, text, metadata
        Score is cosine distance (lower is better)
    """
    collection = get_chroma_collection(package)
    query_vector = embed_query(query)

    result = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
    )

    docs = result["documents"][0]
    metadatas = result["metadatas"][0]
    ids = result["ids"][0]
    distances = result.get("distances", [[None] * len(docs)])[0]

    items = []
    for i, (doc, meta) in enumerate(zip(docs, metadatas)):
        items.append(
            {
                "id": ids[i],
                "score": distances[i],
                "text": doc,
                "metadata": meta,
            }
        )
    return items

def retrieve_bm25(query: str, k: int = 8, package: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieve relevant documents using BM25 keyword matching.
    
    Args:
        query: Query text
        k: Number of results to return
        package: Package name, use default if None
    
    Returns:
        List of retrieval results, each containing id, score, text, metadata
        Score is BM25 score (higher is better)
    """
    if package is None:
        package = DEFAULT_PACKAGE
    
    if package not in PACKAGE_CONFIGS:
        raise ValueError(
            f"Unknown package name: {package}. "
            f"Available packages: {list(PACKAGE_CONFIGS.keys())}"
        )
    
    bm25 = get_bm25_index(package)  # This will auto-rebuild from ChromaDB if needed
    if bm25 is None:
        # BM25 index could not be built (ChromaDB might be empty)
        print(f"[WARNING] BM25 index not available for package '{package}'. Please build vector store first.")
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
        if score > 0:  # Only include chunks with positive BM25 score
            results.append({
                'id': chunk_map[i]['id'],
                'score': score,  # BM25 score (higher is better)
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
                      Higher value = more emphasis on vector similarity results
                      Lower value = more emphasis on BM25 keyword matching
    
    Returns:
        Merged and deduplicated results list with combined scores
    """
    # Normalize scores to [0, 1] range for both channels
    # Vector: distance (lower is better) -> convert to similarity (higher is better)
    # BM25: score (higher is better) -> normalize to [0, 1]
    
    # Normalize vector scores (invert distance to similarity)
    if vector_results:
        vector_scores = [r['score'] for r in vector_results if r['score'] is not None]
        if vector_scores:
            max_vector = max(vector_scores)
            min_vector = min(vector_scores)
            vector_range = max_vector - min_vector if max_vector > min_vector else 1.0
            for r in vector_results:
                if r['score'] is not None:
                    # Convert distance to similarity: (max - score) / range
                    r['normalized_vector_score'] = (max_vector - r['score']) / vector_range
                else:
                    r['normalized_vector_score'] = 0.0
    
    # Normalize BM25 scores
    if bm25_results:
        bm25_scores = [r['score'] for r in bm25_results if r['score'] is not None]
        if bm25_scores:
            max_bm25 = max(bm25_scores)
            min_bm25 = min(bm25_scores)
            bm25_range = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1.0
            for r in bm25_results:
                if r['score'] is not None:
                    r['normalized_bm25_score'] = (r['score'] - min_bm25) / bm25_range
                else:
                    r['normalized_bm25_score'] = 0.0
    
    # Create a map to merge results by ID
    merged_map: Dict[str, Dict[str, Any]] = {}
    
    # Add vector results
    bm25_weight = 1.0 - vector_weight
    for r in vector_results:
        chunk_id = r['id']
        vector_score = r.get('normalized_vector_score', 0.0)
        merged_map[chunk_id] = {
            **r,
            'vector_score': r['score'],  # Original vector distance score
            'bm25_score': None,
            'combined_score': vector_score * vector_weight,
        }
    
    # Merge BM25 results
    for r in bm25_results:
        chunk_id = r['id']
        bm25_score = r.get('normalized_bm25_score', 0.0)
        if chunk_id in merged_map:
            # Update existing entry (found in both channels)
            merged_map[chunk_id]['bm25_score'] = r['score']  # Original BM25 score
            merged_map[chunk_id]['combined_score'] += bm25_score * bm25_weight
        else:
            # New entry from BM25 only
            merged_map[chunk_id] = {
                **r,
                'vector_score': None,
                'bm25_score': r['score'],  # Original BM25 score
                'combined_score': bm25_score * bm25_weight,
            }
    
    # Convert to list and sort by combined score (higher is better)
    merged_results = list(merged_map.values())
    merged_results.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # Use combined_score as the main score for reranking
    # But keep original scores for reference
    for r in merged_results:
        # For reranking, we'll use a normalized distance (lower is better)
        # So convert combined_score (higher is better) to distance (lower is better)
        r['score'] = 1.0 - r['combined_score']  # Convert to distance-like score
    
    return merged_results

def retrieve_multi_packages(query: str, k_per_package: int = 5, packages: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Retrieve from multiple package collections using vector search and merge results.
    
    Args:
        query: Query text
        k_per_package: Number of results per package
        packages: List of packages to query, query all if None
    
    Returns:
        Merged retrieval results, sorted by similarity score (lower distance is better)
    """
    if packages is None:
        packages = list(PACKAGE_CONFIGS.keys())
    
    all_items = []
    for package in packages:
        try:
            items = retrieve_vector(query, k=k_per_package, package=package)
            all_items.extend(items)
        except Exception as e:
            print(f"Warning: Failed to retrieve from package '{package}': {e}")
    
    # Sort by similarity score (smaller distance is better)
    all_items.sort(key=lambda x: x["score"] if x["score"] is not None else float('inf'))
    
    return all_items


def keyword_reranker(
    results: List[Dict[str, Any]], 
    query: str, 
    reranker_weight: float = 0.3,
    html_boost: float = 0.15
) -> List[Dict[str, Any]]:
    """
    Rerank retrieval results based on keyword matching in query and file type.
    Results that explicitly mention query keywords get boosted in ranking.
    HTML files get additional priority boost.
    
    Args:
        results: List of retrieval results (from vector search, BM25, or merged results)
        query: Original query text
        reranker_weight: Weight controlling reranker influence (0.0-1.0)
                        Higher value = stronger keyword boost
        html_boost: Additional boost for HTML files (0.0-1.0)
                   Higher value = stronger HTML priority
    
    Returns:
        Reranked results list with adjusted scores
    """
    import re
    
    # Extract keywords from query (words with 3+ characters, excluding common stop words)
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'how', 'what', 'when', 'where', 'why'}
    query_lower = query.lower()
    # Extract words (alphanumeric sequences)
    words = re.findall(r'\b\w+\b', query_lower)
    keywords = [w for w in words if len(w) >= 3 and w not in stop_words]
    
    # Calculate keyword match scores for each result
    reranked = []
    for item in results:
        text_lower = item['text'].lower()
        title_lower = item['metadata'].get('section_title', '').lower()
        combined_text = f"{title_lower} {text_lower}"
        
        # Count keyword matches using word boundary (exact word match, case-insensitive)
        # This prevents partial matches like "mixed" matching "unmixed" or "remixed"
        matched_keywords = []
        if keywords:
            for keyword in keywords:
                # Use word boundary regex to ensure exact word match
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, combined_text):
                    matched_keywords.append(keyword)
        match_count = len(matched_keywords)
        # Calculate match ratio (how many keywords appear)
        match_ratio = match_count / len(keywords) if keywords else 0
        
        # Check if source file is HTML
        source_file = item['metadata'].get('source_file', '')
        is_html = source_file.lower().endswith('.html')
        
        # Original retrieval score (lower is better for distance-based scores)
        original_score = item['score'] if item['score'] is not None else float('inf')
        
        # Apply reranker boost: reduce score (improve ranking) based on keyword matches
        # reranker_weight controls the strength: 0.0 = no effect, 1.0 = strong effect
        keyword_boost = match_ratio * reranker_weight
        
        # Apply HTML boost: additional priority for HTML files
        html_boost_value = html_boost if is_html else 0.0
        
        # Combine both boosts (total boost cannot exceed 1.0)
        total_boost = min(keyword_boost + html_boost_value, 1.0)
        
        # Reduce distance score proportionally to combined boost
        adjusted_score = original_score * (1.0 - total_boost)
        
        reranked.append({
            **item,
            'score': adjusted_score,
            'original_score': original_score,  # Keep original for reference
            'keyword_matches': match_count,
            'matched_keywords': matched_keywords,  # List of actually matched keywords
            'all_keywords': keywords,  # All extracted keywords from query
            'match_ratio': match_ratio,
            'is_html': is_html,
        })
    
    # Sort by adjusted score (lower is better)
    reranked.sort(key=lambda x: x['score'] if x['score'] is not None else float('inf'))
    
    return reranked

def debug_retrieve(
    package: Optional[str] = None, 
    query: str = "What is this package about?",
    vector_num: int = 30,
    bm25_num: int = 30,
    reranker_num: int = 5,
    html_boost: float = 0.15,
    reranker_weight: float = 0.3,
    vector_weight: float = 0.7,
):
    """
    Test retrieval functionality with dual-channel retrieval (vector similarity + BM25) and keyword reranker.
    
    Args:
        package: Package name
        query: Query text
        vector_num: Number of results from vector similarity search
        bm25_num: Number of results from BM25 keyword search
        reranker_num: Number of final results after reranking
        reranker_weight: Weight controlling reranker influence (0.0-1.0)
                        Higher value = stronger keyword boost
        html_boost: Additional boost for HTML files (0.0-1.0)
                   Higher value = stronger HTML priority
        vector_weight: Weight for vector channel in fusion (0.0-1.0)
                      Higher value = more emphasis on vector similarity results
                      Lower value = more emphasis on BM25 keyword matching
    """
    if package:
        print(f"Retrieving from package: {package}")
        print(f"Query: {query}")
        print(f"Dual-channel retrieval: {vector_num} vector results, {bm25_num} BM25 results (vector_weight={vector_weight})")
        print(f"After reranking: {reranker_num} results (keyword_weight={reranker_weight}, html_boost={html_boost})")
        print()
        
        # Step 1: Dual-channel retrieval
        # Channel 1: Vector similarity retrieval
        vector_results = retrieve_vector(query, k=vector_num, package=package)
        print(f"[DEBUG] Vector channel: {len(vector_results)} results")
        
        # Channel 2: BM25 keyword retrieval
        bm25_results = retrieve_bm25(query, k=bm25_num, package=package)
        print(f"[DEBUG] BM25 channel: {len(bm25_results)} results")
        
        # Step 2: Merge results from both channels
        merged_results = merge_retrieval_results(
            vector_results, 
            bm25_results, 
            vector_weight=vector_weight
        )
        print(f"[DEBUG] Merged results: {len(merged_results)} unique chunks")
        print()
        
        # Step 3: Keyword reranking with HTML priority
        reranked_results = keyword_reranker(merged_results, query, reranker_weight, html_boost)
        
        # Step 4: Return top reranker_num results
        final_results = reranked_results[:reranker_num]
        
        # Display results
        for i, r in enumerate(final_results, 1):
            package_name = r['metadata'].get('package', 'unknown')
            original_score = r.get('original_score', r['score'])
            keyword_matches = r.get('keyword_matches', 0)
            match_ratio = r.get('match_ratio', 0)
            is_html = r.get('is_html', False)
            file_type = "HTML" if is_html else "Other"
            
            matched_keywords = r.get('matched_keywords', [])
            all_keywords = r.get('all_keywords', [])
            
            # Show channel scores if available
            vector_score = r.get('vector_score')
            bm25_score = r.get('bm25_score')
            combined_score = r.get('combined_score')
            
            print(f"=== Result {i} (package={package_name}, type={file_type}) ===")
            if vector_score is not None and bm25_score is not None:
                print(f"Vector score: {vector_score:.4f} | BM25 score: {bm25_score:.4f} | Combined: {combined_score:.4f}")
            elif vector_score is not None:
                print(f"Vector score: {vector_score:.4f} | BM25 score: N/A")
            elif bm25_score is not None:
                print(f"Vector score: N/A | BM25 score: {bm25_score:.4f}")
            print(f"Final reranked score: {r['score']:.4f}")
            print(f"Query keywords: {all_keywords}")
            print(f"Matched keywords: {matched_keywords} ({keyword_matches}/{len(all_keywords) if all_keywords else 0} = {match_ratio*100:.1f}%)")
            print(f"Source: {r['metadata']['source_file']} - {r['metadata']['section_title']}")
            print(r["text"][:1500])
            print()
    else:
        raise ValueError("Package is required")





# Entry point
if __name__ == "__main__":
    # Preload embedding model to avoid delay on first query
    preload_embed_model()
    
    # ====== Configuration ======
    # Set to True ONLY if:
    #   1. First time building the database, OR
    #   2. Document files have been updated and you want to refresh the database
    # Set to False if ChromaDB already contains your documents
    # Note: BM25 index will be automatically rebuilt from ChromaDB on first use
    REBUILD_VECTOR_STORE = False  # Set to True only if you need to rebuild the database
    
    # ====== Main execution ======
    if REBUILD_VECTOR_STORE:
        print("=" * 60)
        print("Building vector store (vector embeddings + BM25 index)...")
        print("=" * 60)
        build_vector_store(package="itensormps")
        print("\n" + "=" * 60)
        print("Vector store built successfully!")
        print("Set REBUILD_VECTOR_STORE = False to skip rebuilding next time.")
        print("=" * 60 + "\n")
    else:
        print("Using existing ChromaDB database")
        print("Note: BM25 index will be automatically rebuilt from ChromaDB on first retrieval.\n")
    
    # Run retrieval test
    debug_retrieve(
        package="itensormps", 
        query="How to calculate the energy of the excited state using DMRG?",
        vector_num=50,
        bm25_num=50,
        reranker_num=5,
        html_boost=0.15,
        reranker_weight=0.5,
        vector_weight=0.5,  # Weight for vector channel (0.7 = 70% vector, 30% BM25)
    )


