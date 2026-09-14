"""Vector store building functions for RAG retrieval.

This module contains all functions for building ChromaDB vector stores from
authority_library documents. These functions are used by the build_vector_store.py
entry point script.
"""

import os
import platform
import sys
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

# Auto-configure huggingface_hub symlinks warning based on OS
if platform.system() == "Windows":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
else:
    pass

# Configure huggingface_hub download timeout
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

import tiktoken
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

from mcp_server_retrieve.retrieval import (
    find_project_root,
    set_project_root,
    get_package_configs,
    get_chroma_client,
    get_chroma_collection,
    build_bm25_index,
)


# ========= Chunk Definition =========
@dataclass
class Chunk:
    id: str
    text: str
    source_file: str
    section_title: str
    section_id: str


# ========= Embedding Functions =========
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
_embed_model: Optional[SentenceTransformer] = None


def _get_retrieve_mcp_root() -> Path:
    """
    Get the retrieve-mcp root directory (where this package is located).
    This is used to locate the local model storage directory.
    """
    # This file is in: retrieve-mcp/src/mcp_server_retrieve/vector_store_builder.py
    # So retrieve-mcp root is: ../../ from this file
    current_file = Path(__file__).resolve()
    retrieve_mcp_root = current_file.parent.parent.parent
    return retrieve_mcp_root


def _get_local_model_path() -> Path:
    """
    Get the local path where the embedding model should be stored.
    Model is stored in: retrieve-mcp/models/embedding_model/
    """
    retrieve_mcp_root = _get_retrieve_mcp_root()
    model_dir = retrieve_mcp_root / "models" / "embedding_model"
    return model_dir


def get_embed_model() -> SentenceTransformer:
    """
    Lazy load sentence-transformers model.
    
    Priority order:
    1. Load from local project directory (retrieve-mcp/models/embedding_model/)
    2. If not found locally, download from Hugging Face to local directory
    3. If download fails, fall back to Hugging Face cache (default behavior)
    """
    global _embed_model
    if _embed_model is None:
        local_model_path = _get_local_model_path()
        
        # Try to load from local project directory first
        if local_model_path.exists() and (local_model_path / "config.json").exists():
            print(f"[INFO] Loading embedding model from local directory: {local_model_path}")
            _embed_model = SentenceTransformer(str(local_model_path))
            print(f"[INFO] Model loaded successfully from local directory!")
        else:
            # Model not found locally, try to download to local directory
            print(f"[INFO] Model not found in local directory. Downloading to: {local_model_path}")
            print(f"[INFO] Downloading model: {EMBED_MODEL_NAME} from Hugging Face...")
            
            try:
                # Download model to local directory
                local_model_path.parent.mkdir(parents=True, exist_ok=True)
                _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
                # Save to local directory
                _embed_model.save(str(local_model_path))
                print(f"[INFO] Model downloaded and saved to: {local_model_path}")
                print(f"[INFO] Future runs will use this local copy.")
            except Exception as download_error:
                # If download/save fails, fall back to using Hugging Face cache
                print(f"[WARNING] Failed to download/save model to local directory: {download_error}")
                print(f"[WARNING] Falling back to Hugging Face cache (default behavior).")
                _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
                print(f"[INFO] Model loaded from Hugging Face cache.")
    return _embed_model


def embed_passages(texts: List[str]) -> List[List[float]]:
    """Embed document chunks (passages)."""
    model = get_embed_model()
    prefixed = [f"passage: {t}" for t in texts]
    emb = model.encode(prefixed, normalize_embeddings=True)
    return [e.tolist() for e in emb]


# ========= Token Counting =========
enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Estimate token count for text."""
    return len(enc.encode(text))


def split_long_text(text: str, max_tokens: int = 600, overlap_tokens: int = 100) -> List[str]:
    """Split long text into chunks by token length with overlap."""
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
        start = end - overlap_tokens

    return chunks


# ========= Chunk Extraction Functions =========
def generate_unique_chunk_id(file_path: Path, section_id: str, chunk_index: int) -> str:
    """Generate unique chunk ID based on file absolute path."""
    import hashlib
    abs_path = str(file_path.resolve())
    path_hash = hashlib.md5(abs_path.encode('utf-8')).hexdigest()
    file_name = file_path.stem
    return f"{file_name}__{path_hash}__{section_id}__{chunk_index}"


def extract_sections_from_html(html_path: Path, doc_root: Path) -> List[Chunk]:
    """Extract structured document chunks from HTML file."""
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")

    # Remove navigation/sidebar
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

    for el in body.descendants:
        if not getattr(el, "name", None):
            continue

        if el.name == "h2":
            chunks.extend(flush_buffer())
            current_title = el.get_text(strip=True) or "UNTITLED"
            current_section_id = el.get("id") or f"sec_{sec_index}"
            if current_title and current_title != "UNTITLED":
                buffer_lines.append(f"## {current_title}")
        elif el.name in {"p", "li"}:
            txt = el.get_text(" ", strip=True)
            if txt:
                buffer_lines.append(txt)
        elif el.name == "pre":
            code_txt = el.get_text("\n", strip=True)
            if code_txt:
                buffer_lines.append("```code\n" + code_txt + "\n```")

    chunks.extend(flush_buffer())
    return chunks


def extract_sections_from_code(code_path: Path, doc_root: Path) -> List[Chunk]:
    """Extract document chunks from code files."""
    try:
        content = code_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[WARNING] Failed to read {code_path}: {e}")
        return []

    if not content.strip():
        return []

    chunks: List[Chunk] = []
    lines = content.split('\n')

    if len(lines) > 1000:
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
    """Extract document chunks from Markdown file."""
    try:
        content = md_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[WARNING] Failed to read {md_path}: {e}")
        return []

    if not content.strip():
        return []

    chunks: List[Chunk] = []
    lines = content.split('\n')
    current_title = md_path.stem
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
        if stripped.startswith('#'):
            chunks.extend(flush_buffer())
            level = len(line) - len(line.lstrip('#'))
            title_text = stripped.lstrip('#').strip()
            if title_text:
                current_title = title_text
                current_section_id = f"h{level}_{sec_index}"
            else:
                current_title = "UNTITLED"
                current_section_id = f"h{level}_{sec_index}"
        else:
            if stripped:
                buffer_lines.append(line)

    chunks.extend(flush_buffer())
    return chunks


def extract_sections_from_text(txt_path: Path, doc_root: Path) -> List[Chunk]:
    """Extract document chunks from plain text file."""
    try:
        content = txt_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[WARNING] Failed to read {txt_path}: {e}")
        return []

    if not content.strip():
        return []

    # XYZ structure files: keep entire file as one chunk (no splitting) so retrieval returns complete structures
    if txt_path.suffix.lower() == ".xyz":
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


def collect_all_chunks(doc_root: Path, file_extensions: List[str]) -> List[Chunk]:
    """Collect document chunks from all specified file types in directory."""
    all_chunks: List[Chunk] = []
    all_files: List[Path] = []

    for ext in file_extensions:
        pattern = f"**/*{ext}"
        files = list(doc_root.glob(pattern))
        all_files.extend(files)
        print(f"Found {len(files)} {ext} files")

    excluded_names = {"search.html", "__pycache__", ".git", ".svn", ".hg"}
    filtered_files = [
        f for f in all_files
        if f.name.lower() not in excluded_names
        and not any(excluded in str(f) for excluded in excluded_names)
    ]

    print(f"Total files to process: {len(filtered_files)}")

    for path in filtered_files:
        ext = path.suffix.lower()
        try:
            if ext == ".html":
                file_chunks = extract_sections_from_html(path, doc_root)
            elif ext in [".py", ".jl", ".js", ".ts", ".cpp", ".c", ".h", ".hpp", ".java", ".go", ".rs"]:
                file_chunks = extract_sections_from_code(path, doc_root)
            elif ext in [".md", ".markdown"]:
                file_chunks = extract_sections_from_markdown(path, doc_root)
            elif ext in [".txt", ".text", ".xyz"]:
                file_chunks = extract_sections_from_text(path, doc_root)
            else:
                print(f"[WARNING] Unknown file type {ext}, treating as text: {path}")
                file_chunks = extract_sections_from_text(path, doc_root)

            all_chunks.extend(file_chunks)
        except Exception as e:
            print(f"[ERROR] Failed to process {path}: {e}")
            continue

    print(f"Total chunks extracted: {len(all_chunks)}")
    return all_chunks


# ========= Build Vector Store =========
def build_vector_store(package: str, chroma_path: Optional[str] = None, project_root: Optional[Path] = None):
    """
    Build vector database for specified package.
    
    Args:
        package: Package name (e.g. "itensormps", "itensors", "netket")
        chroma_path: ChromaDB storage path (relative to project root)
        project_root: Project root directory (auto-detected if not provided)
    """
    # Set project root
    if project_root:
        set_project_root(project_root)
    else:
        try:
            detected_root = find_project_root()
            set_project_root(detected_root)
            project_root = detected_root
        except Exception as e:
            print(f"[ERROR] Failed to auto-detect project root: {e}")
            print("[ERROR] Please provide --project-root argument")
            sys.exit(1)

    # Get package configuration from retrieval.py
    # This includes doc_root, collection_name, and file_extensions
    # The file_extensions list determines which file types will be processed
    configs = get_package_configs()
    if package not in configs:
        raise ValueError(
            f"Unknown package name: {package}. "
            f"Available packages: {list(configs.keys())}"
        )

    config = configs[package]
    doc_root = config["doc_root"]
    # file_extensions is read from retrieval.py get_package_configs() function
    # To modify which file types are processed, edit retrieval.py lines 114-129
    file_extensions = config.get("file_extensions", [".html"])

    # Resolve chroma_path relative to project root
    if chroma_path is None:
        chroma_path = "authority_library/rag_store/rag_chroma"
    
    chroma_path_obj = Path(chroma_path)
    if not chroma_path_obj.is_absolute():
        chroma_path = str(project_root / chroma_path_obj)
    else:
        chroma_path = str(chroma_path_obj.resolve())

    print(f"[INFO] Step 1/4: Clearing and getting collection '{config['collection_name']}'...")
    print(f"[INFO] ChromaDB path: {chroma_path}")
    client = get_chroma_client(chroma_path=chroma_path)
    try:
        client.delete_collection(config["collection_name"])
        print(f"[INFO] Deleted existing collection for rebuild")
    except Exception as e:
        print(f"[INFO] No existing collection to delete (first build or: {e})")
    collection = get_chroma_collection(package, chroma_path)

    print(f"[INFO] Step 2/4: Collecting chunks from '{doc_root}' (file types: {file_extensions})...")
    chunks = collect_all_chunks(doc_root, file_extensions=file_extensions)
    
    print(f"[INFO] Step 2.5/4: Building BM25 index for {len(chunks)} chunks...")
    build_bm25_index(package, chunks, chroma_path)
    
    print(f"[INFO] Step 3/4: Generating vector embeddings for {len(chunks)} chunks...")

    batch_size = 64
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c.text for c in batch]
        metas = [
            {
                "package": package,
                "source_file": c.source_file,
                "section_title": c.section_title,
                "section_id": c.section_id,
            }
            for c in batch
        ]
        ids = [c.id for c in batch]

        batch_num = i // batch_size + 1
        if batch_num % 5 == 0 or batch_num == total_batches:
            print(f"[INFO] Processing batch {batch_num}/{total_batches}...")

        embeddings = embed_passages(texts)
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metas,
        )

    print(f"[INFO] Step 4/4: Done! Inserted {len(chunks)} chunks.")

