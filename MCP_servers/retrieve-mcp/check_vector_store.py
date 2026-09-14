"""Check vector store statistics - Entry Point.

This script allows users to check the number of chunks and file types
in existing vector stores for each package.

Usage:
    python check_vector_store.py [--package PACKAGE] [--chroma-path PATH]
"""

import argparse
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, Set

# Import functions from the module
sys.path.insert(0, str(Path(__file__).parent / "src"))
import chromadb
from mcp_server_retrieve.retrieval import (
    get_chroma_collection,
    get_package_configs,
    find_project_root,
    set_project_root,
)


def get_file_extensions_from_metadata_data(all_data: dict) -> Dict[str, int]:
    """
    Extract file extensions from already-fetched collection data.
    
    Args:
        all_data: Dictionary from collection.get() containing ids, documents, metadatas
        
    Returns:
        Dictionary mapping file extensions to chunk counts
    """
    try:
        # Check if we have metadatas
        metadatas = all_data.get("metadatas", [])
        if not metadatas:
            return {}
        
        # Extract file extensions from source_file metadata
        extensions = []
        for metadata in metadatas:
            if metadata and isinstance(metadata, dict):
                source_file = metadata.get("source_file", "")
                if source_file:
                    ext = Path(source_file).suffix.lower()
                    if ext:  # Only count files with extensions
                        extensions.append(ext)
        
        # Count extensions
        return dict(Counter(extensions))
    except Exception as e:
        print(f"[WARNING] Failed to extract file types from data: {e}")
        return {}


def get_file_extensions_from_metadata(collection) -> Dict[str, int]:
    """
    Extract file extensions from collection metadata and count them.
    
    Args:
        collection: ChromaDB collection object
        
    Returns:
        Dictionary mapping file extensions to chunk counts
    """
    try:
        # Get all data from collection
        # ChromaDB's get() method without arguments returns all items
        result = collection.get()
        
        if not result:
            return {}
        
        return get_file_extensions_from_metadata_data(result)
    except Exception as e:
        print(f"[WARNING] Failed to extract file types: {e}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return {}


def get_chunk_count_from_sqlite(chroma_path: str, collection_name: str) -> int:
    """
    Get chunk count by directly querying ChromaDB's SQLite database.
    Note: In newer ChromaDB versions, data may be stored in HNSW files, not SQLite.
    
    Args:
        chroma_path: Path to ChromaDB storage directory
        collection_name: Name of the collection
        
    Returns:
        Number of chunks in the collection, or -1 if query failed
    """
    try:
        import sqlite3
        sqlite_path = Path(chroma_path) / "chroma.sqlite3"
        
        if not sqlite_path.exists():
            return -1
        
        conn = sqlite3.connect(str(sqlite_path))
        cursor = conn.cursor()
        
        # Query to get collection ID from name
        cursor.execute("SELECT id FROM collections WHERE name = ?", (collection_name,))
        collection_result = cursor.fetchone()
        
        if not collection_result:
            conn.close()
            return -1
        
        collection_id = collection_result[0]
        
        # In newer ChromaDB, embeddings are stored in segments, not directly in embeddings table
        # Try to count through segments
        # First, find metadata segments for this collection
        cursor.execute("""
            SELECT COUNT(*) 
            FROM segments s
            WHERE s.collection = ? AND s.type LIKE '%metadata%'
        """, (collection_id,))
        
        # Actually, let's try a different approach - check if there's a way to count
        # through the segment metadata or check the HNSW index files
        # For now, return -1 to indicate we need to use other methods
        conn.close()
        return -1  # Data is likely in HNSW files, not SQLite
    except Exception as e:
        print(f"[DEBUG] SQLite query failed: {e}")
        return -1


def check_package_stats(package: str, chroma_path: str, project_root: Path, debug: bool = False) -> Dict:
    """
    Check statistics for a single package using multiple methods.
    Uses the same functions as retrieval.py to ensure consistency.
    
    Args:
        package: Package name
        chroma_path: ChromaDB storage path (relative to project root)
        project_root: Project root directory
        
    Returns:
        Dictionary with package statistics
    """
    try:
        configs = get_package_configs()
        
        if package not in configs:
            return {
                "package": package,
                "chunk_count": 0,
                "file_types": {},
                "status": "error",
                "error": f"Unknown package: {package}"
            }
        
        collection_name = configs[package]["collection_name"]
        
        # IMPORTANT: Calculate absolute path and use it directly
        # get_chroma_client uses singleton and may have wrong path cached
        chroma_path_abs = Path(chroma_path)
        if not chroma_path_abs.is_absolute():
            chroma_path_abs = project_root / chroma_path_abs
        else:
            chroma_path_abs = chroma_path_abs.resolve()
        
        if debug:
            print(f"[DEBUG] Package {package}: Absolute ChromaDB path: {chroma_path_abs}")
            print(f"[DEBUG] Package {package}: Path exists: {chroma_path_abs.exists()}")
        
        # Use direct client creation with absolute path to avoid singleton issues
        # CRITICAL: Use get_collection() NOT get_or_create_collection() to avoid creating new collections
        try:
            client = chromadb.PersistentClient(path=str(chroma_path_abs))
            
            # Check if collection exists first
            existing_collections = client.list_collections()
            collection_exists = any(col.name == collection_name for col in existing_collections)
            
            if not collection_exists:
                if debug:
                    print(f"[DEBUG] Package {package}: Collection '{collection_name}' does not exist, skipping...")
                return {
                    "package": package,
                    "collection_name": collection_name,
                    "chunk_count": 0,
                    "file_types": {},
                    "status": "not_found",
                    "error": f"Collection '{collection_name}' does not exist in ChromaDB"
                }
            
            # Only get existing collection, do NOT create
            collection = client.get_collection(name=collection_name)
            if debug:
                print(f"[DEBUG] Package {package}: Got existing collection '{collection_name}' using direct client")
        except Exception as e:
            # If get_collection fails, the collection doesn't exist
            if "does not exist" in str(e).lower() or "not found" in str(e).lower():
                if debug:
                    print(f"[DEBUG] Package {package}: Collection '{collection_name}' does not exist: {e}")
                return {
                    "package": package,
                    "collection_name": collection_name,
                    "chunk_count": 0,
                    "file_types": {},
                    "status": "not_found",
                    "error": f"Collection '{collection_name}' does not exist in ChromaDB"
                }
            else:
                return {
                    "package": package,
                    "collection_name": collection_name,
                    "chunk_count": 0,
                    "file_types": {},
                    "status": "error",
                    "error": f"Failed to access collection '{collection_name}': {str(e)}"
                }
        
        chunk_count = 0
        method_used = "unknown"
        all_data = None  # Store data for file type extraction
        
        # Method 1: Try count() method
        try:
            chunk_count = collection.count()
            if debug:
                print(f"[DEBUG] Package {package}: count() returned: {chunk_count}")
            if chunk_count > 0:
                method_used = "count()"
                if debug:
                    print(f"[INFO] Package {package}: Using count() method, found {chunk_count} chunks")
        except Exception as e:
            if debug:
                print(f"[DEBUG] Package {package}: count() failed: {e}")
                import traceback
                print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        
        # Method 1.5: Try get() method (same as rebuild_bm25_from_chromadb uses)
        if chunk_count == 0:
            try:
                if debug:
                    print(f"[DEBUG] Package {package}: Trying collection.get()...")
                all_data = collection.get()
                if debug:
                    print(f"[DEBUG] Package {package}: get() returned keys: {list(all_data.keys()) if all_data else 'None'}")
                if all_data:
                    ids = all_data.get('ids', [])
                    documents = all_data.get('documents', [])
                    metadatas = all_data.get('metadatas', [])
                    if debug:
                        print(f"[DEBUG] Package {package}: ids={len(ids)}, docs={len(documents)}, metas={len(metadatas)}")
                    if ids and len(ids) > 0:
                        chunk_count = len(ids)
                        method_used = "get()"
                        if debug:
                            print(f"[INFO] Package {package}: Using get() method, found {chunk_count} chunks")
            except Exception as e:
                if debug:
                    print(f"[DEBUG] Package {package}: get() failed: {e}")
                    import traceback
                    print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        
        # Method 2: Try using query() with a dummy embedding to get all results
        # This works by querying with a zero vector and requesting a very large number of results
        if chunk_count == 0:
            try:
                # Get embedding dimension from collection metadata or use default
                # bge-base-en-v1.5 has 768 dimensions
                embedding_dim = 768
                try:
                    # Try to get dimension from collection
                    coll_meta = collection.metadata
                    if coll_meta and 'hnsw:space' in str(coll_meta):
                        # Try to infer from collection, but use 768 as default
                        pass
                except:
                    pass
                
                # Create a zero vector for querying
                dummy_embedding = [0.0] * embedding_dim
                
                # Query with very large n_results to get all documents
                query_result = collection.query(
                    query_embeddings=[dummy_embedding],
                    n_results=100000  # Very large number
                )
                
                if query_result and query_result.get("ids") and len(query_result["ids"]) > 0:
                    chunk_count = len(query_result["ids"][0])
                    if chunk_count > 0:
                        method_used = "query() with dummy vector"
                        if debug:
                            print(f"[INFO] Package {package}: Using query() method, found {chunk_count} chunks")
            except Exception as e:
                if debug:
                    print(f"[DEBUG] Package {package}: query() method failed: {e}")
        
        # Method 3: Try get() with different parameters
        if chunk_count == 0:
            try:
                # Try get() without any parameters (should return all)
                all_data = collection.get()
                ids = all_data.get("ids", [])
                if ids and len(ids) > 0:
                    chunk_count = len(ids)
                    method_used = "get()"
                    if debug:
                        print(f"[INFO] Package {package}: Using get(), found {chunk_count} chunks")
            except Exception as e:
                if debug:
                    print(f"[DEBUG] Package {package}: get() failed: {e}")
        
        # If we still don't have data but have chunk_count > 0, try to get data for file type extraction
        if chunk_count > 0 and all_data is None:
            try:
                if debug:
                    print(f"[DEBUG] Package {package}: Fetching data for file type extraction...")
                all_data = collection.get()
            except Exception as e:
                if debug:
                    print(f"[DEBUG] Package {package}: Failed to fetch data for file types: {e}")
                all_data = None
        
        # Method 4: Try peek() to see if there's any data
        if chunk_count == 0:
            try:
                peek_data = collection.peek(limit=1000)
                if peek_data:
                    ids = peek_data.get("ids", [])
                    if ids and len(ids) > 0:
                        # If peek returns data, the collection has data
                        # Try to get more by using a larger limit or query
                        chunk_count = len(ids)
                        method_used = f"peek(limit=1000) - showing first {chunk_count}"
                        if debug:
                            print(f"[INFO] Package {package}: Using peek(), found at least {chunk_count} chunks (may be more)")
            except Exception as e:
                if debug:
                    print(f"[DEBUG] Package {package}: peek() failed: {e}")
        
        # Get file types from metadata
        file_types = {}
        if all_data is not None:
            try:
                file_types = get_file_extensions_from_metadata_data(all_data)
                if debug:
                    print(f"[DEBUG] Package {package}: Extracted {len(file_types)} file types")
            except Exception as e:
                if debug:
                    print(f"[DEBUG] Package {package}: Failed to extract file types: {e}")
                file_types = {}
        elif chunk_count > 0:
            # If we have chunks but no data, try to get file types directly
            try:
                if debug:
                    print(f"[DEBUG] Package {package}: Attempting to extract file types from collection...")
                file_types = get_file_extensions_from_metadata(collection)
                if debug:
                    print(f"[DEBUG] Package {package}: Extracted {len(file_types)} file types")
            except Exception as e:
                if debug:
                    print(f"[DEBUG] Package {package}: Failed to extract file types: {e}")
                file_types = {}
        
        return {
            "package": package,
            "collection_name": collection_name,
            "chunk_count": chunk_count,
            "file_types": file_types,
            "method_used": method_used,
            "status": "success" if chunk_count > 0 else "empty"
        }
    except Exception as e:
        import traceback
        return {
            "package": package,
            "chunk_count": 0,
            "file_types": {},
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check vector store statistics (chunks count and file types) for packages"
    )
    parser.add_argument(
        "--package",
        type=str,
        default=None,
        help="Package name to check (e.g. itensormps, itensors, netket, qiskit, qiskit-nature, qiskit-algorithms, qiskit-aer). If not specified, checks all packages.",
    )
    parser.add_argument(
        "--chroma-path",
        type=str,
        default="authority_library/rag_store/rag_chroma",
        help="ChromaDB storage path (relative to project root, default: authority_library/rag_store/rag_chroma)",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root directory (auto-detected if not provided)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable detailed debug output (default: False)",
    )
    
    args = parser.parse_args()

    # Set project root
    # CRITICAL: Script is at project_root/MCP_servers/retrieve-mcp/check_vector_store.py
    # We need to go up 2 levels from script location to find project_root
    if args.project_root:
        project_root = Path(args.project_root).resolve()
        set_project_root(project_root)
    else:
        try:
            # Get script file location
            script_file = Path(__file__).resolve()
            if args.debug:
                print(f"[DEBUG] Script file: {script_file}")
            
            # Script is at: project_root/MCP_servers/retrieve-mcp/check_vector_store.py
            # So project_root = script_file.parent.parent.parent (go up 3 levels)
            script_dir = script_file.parent  # MCP_servers/retrieve-mcp
            mcp_servers_dir = script_dir.parent  # MCP_servers
            project_root_candidate = mcp_servers_dir.parent  # project_root
            
            if args.debug:
                print(f"[DEBUG] Calculated project root candidate: {project_root_candidate}")
            
            # Verify this is the correct project root by checking for authority_library
            authority_lib = project_root_candidate / "authority_library"
            if args.debug:
                print(f"[DEBUG] Checking authority_library at: {authority_lib}")
                print(f"[DEBUG] Exists: {authority_lib.exists()}, Is dir: {authority_lib.is_dir() if authority_lib.exists() else 'N/A'}")
            
            if authority_lib.exists() and authority_lib.is_dir():
                detected_root = project_root_candidate
                if args.debug:
                    print(f"[DEBUG] Using calculated project root: {detected_root}")
            else:
                # If calculated path doesn't work, search upward from script directory
                if args.debug:
                    print(f"[DEBUG] Calculated path invalid, searching upward...")
                current = script_dir
                detected_root = None
                max_levels = 10  # Safety limit
                level = 0
                while current != current.parent and level < max_levels:
                    authority_lib = current / "authority_library"
                    if authority_lib.exists() and authority_lib.is_dir():
                        detected_root = current
                        if args.debug:
                            print(f"[DEBUG] Found project root at level {level}: {detected_root}")
                        break
                    current = current.parent
                    level += 1
                
                if detected_root is None:
                    # Last resort: try find_project_root()
                    if args.debug:
                        print(f"[DEBUG] Upward search failed, trying find_project_root()...")
                    detected_root = find_project_root()
                    if args.debug:
                        print(f"[DEBUG] find_project_root() returned: {detected_root}")
            
            # CRITICAL VERIFICATION
            authority_lib = detected_root / "authority_library"
            if not authority_lib.exists() or not authority_lib.is_dir():
                raise FileNotFoundError(
                    f"Could not find authority_library directory.\n"
                    f"  Script location: {script_file}\n"
                    f"  Detected root: {detected_root}\n"
                    f"  Current working directory: {Path.cwd()}\n"
                    f"  Calculated candidate: {project_root_candidate}\n"
                    f"  Expected: {script_file.parent.parent.parent} / authority_library"
                )
            
            project_root = detected_root
            set_project_root(project_root)
            if args.debug:
                print(f"[INFO] Project root: {project_root}")
            
            # CRITICAL: Verify the chroma path exists
            chroma_path_test = project_root / args.chroma_path
            if not chroma_path_test.exists():
                raise FileNotFoundError(
                    f"ChromaDB path does not exist: {chroma_path_test}\n"
                    f"Expected: {project_root} / {args.chroma_path}\n"
                    f"Please verify the path is correct."
                )
            if args.debug:
                print(f"[INFO] Verified ChromaDB path exists: {chroma_path_test}")
        except Exception as e:
            print(f"[ERROR] Failed to detect project root: {e}")
            print(f"[ERROR] Script location: {Path(__file__).resolve()}")
            print(f"[ERROR] Current working directory: {Path.cwd()}")
            print("[ERROR] Please provide --project-root argument explicitly")
            print(f"[ERROR] Example: --project-root \"<path-to-project-root>\"")
            sys.exit(1)

    # Get package configurations
    configs = get_package_configs()
    
    # Determine which packages to check
    if args.package:
        if args.package not in configs:
            print(f"[ERROR] Unknown package: {args.package}")
            print(f"Available packages: {list(configs.keys())}")
            sys.exit(1)
        packages_to_check = [args.package]
    else:
        packages_to_check = list(configs.keys())

    print("=" * 70)
    print("Vector Store Statistics")
    print("=" * 70)
    if args.debug:
        print(f"ChromaDB path: {args.chroma_path}")
        print(f"Project root: {project_root}")
        print()

    # Use relative path (as retrieval.py expects)
    # retrieval.py's get_chroma_client expects relative path from project root
    chroma_path = args.chroma_path  # Keep as relative path
    
    # For debugging, show absolute path
    chroma_path_abs = Path(chroma_path)
    if not chroma_path_abs.is_absolute():
        chroma_path_abs = project_root / chroma_path_abs
    else:
        chroma_path_abs = chroma_path_abs.resolve()
    
    if args.debug:
        print(f"[INFO] ChromaDB path (relative): {chroma_path}")
        print(f"[INFO] ChromaDB path (absolute): {chroma_path_abs}")
    
    # List all existing collections for information (use absolute path for direct client)
    try:
        client = chromadb.PersistentClient(path=str(chroma_path_abs))
        existing_collections = client.list_collections()
        collection_names = [col.name for col in existing_collections]
        if args.debug:
            print(f"[INFO] Existing collections in ChromaDB: {collection_names}")
            print()
        else:
            # Show summary even without debug
            print(f"Found {len(collection_names)} existing collection(s) in ChromaDB")
            print()
    except Exception as e:
        if args.debug:
            print(f"[WARNING] Could not list collections: {e}")
            print()

    # Check each package
    # Pass relative path to check_package_stats, which will use get_chroma_collection
    all_stats = []
    for package in packages_to_check:
        if args.debug:
            print(f"[INFO] Checking package: {package}...")
        stats = check_package_stats(package, chroma_path, project_root, debug=args.debug)
        all_stats.append(stats)

    # Display results
    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)
    
    total_chunks = 0
    for stats in all_stats:
        print()
        print(f"Package: {stats['package']}")
        print("-" * 70)
        
        if stats['status'] == 'error':
            print(f"  Status: ERROR - {stats.get('error', 'Unknown error')}")
            if 'traceback' in stats:
                print(f"  Details: {stats['traceback']}")
            print(f"  Chunks: 0")
            print(f"  File types: None")
        elif stats['status'] == 'not_found':
            # Collection does not exist - this is expected for some packages
            if 'collection_name' in stats:
                print(f"  Collection name: {stats['collection_name']}")
            print(f"  Status: Collection does not exist (not built yet)")
            print(f"  Chunks: 0")
            print(f"  File types: N/A")
        else:
            # Success case - display all information
            if 'collection_name' in stats:
                print(f"  Collection name: {stats['collection_name']}")
            
            chunk_count = stats['chunk_count']
            print(f"  Chunks: {chunk_count:,}")
            
            if 'method_used' in stats and args.debug:
                print(f"  Method used: {stats['method_used']}")
            
            if stats['status'] == 'empty':
                print(f"  Status: Collection appears to be empty")
            else:
                total_chunks += chunk_count
            
            # Display file types
            if stats['file_types']:
                print(f"  File types ({len(stats['file_types'])} types):")
                # Sort by count (descending)
                sorted_types = sorted(
                    stats['file_types'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                for ext, count in sorted_types:
                    percentage = (count / chunk_count) * 100 if chunk_count > 0 else 0
                    print(f"    {ext:10s}: {count:6,} chunks ({percentage:5.1f}%)")
            else:
                if chunk_count > 0:
                    print(f"  File types: Unable to extract metadata (chunks exist but metadata unavailable)")
                else:
                    print(f"  File types: N/A (collection is empty)")

    print()
    print("=" * 70)
    print(f"Total chunks across all packages: {total_chunks:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()

