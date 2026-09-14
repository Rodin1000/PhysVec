"""Delete ChromaDB collection - Entry Point.

This script allows users to delete a specific collection from ChromaDB.
Use with caution - this operation cannot be undone.

Usage:
    python delete_collection.py --package netket [--confirm]
"""

import argparse
import sys
from pathlib import Path

# Import functions from the module
sys.path.insert(0, str(Path(__file__).parent / "src"))
import chromadb
from mcp_server_retrieve.retrieval import (
    get_package_configs,
    find_project_root,
    set_project_root,
)


def delete_collection(package: str, chroma_path: str, project_root: Path, confirm: bool = False) -> bool:
    """
    Delete a ChromaDB collection for a specific package.
    
    Args:
        package: Package name (e.g. "netket")
        chroma_path: ChromaDB storage path (relative to project root)
        project_root: Project root directory
        confirm: If True, skip confirmation prompt
        
    Returns:
        True if deletion was successful, False otherwise
    """
    try:
        # Get package configs - this should NOT create any collections
        # get_package_configs() only reads configuration, doesn't touch ChromaDB
        # It calls get_project_root() which may trigger initialization, but shouldn't create collections
        configs = get_package_configs()
        
        if package not in configs:
            print(f"[ERROR] Unknown package: {package}")
            print(f"Available packages: {list(configs.keys())}")
            return False
        
        collection_name = configs[package]["collection_name"]
        
        # CRITICAL: At this point, we have NOT connected to ChromaDB yet
        # So no collections can have been created
        
        # Calculate absolute path
        chroma_path_abs = Path(chroma_path)
        if not chroma_path_abs.is_absolute():
            chroma_path_abs = project_root / chroma_path_abs
        else:
            chroma_path_abs = chroma_path_abs.resolve()
        
        print(f"[INFO] ChromaDB path: {chroma_path_abs}")
        print(f"[INFO] Package: {package}")
        print(f"[INFO] Collection name: {collection_name}")
        
        # Connect to ChromaDB
        client = chromadb.PersistentClient(path=str(chroma_path_abs))
        
        # CRITICAL: Check if collection exists FIRST, before any operations
        # This prevents any accidental collection creation
        existing_collections = client.list_collections()
        collection_names = [col.name for col in existing_collections]
        collection_exists = collection_name in collection_names
        
        # CRITICAL: If collection does not exist, return immediately
        # Do NOT perform any operations that might create collections
        if not collection_exists:
            print(f"[INFO] Collection '{collection_name}' does not exist. Nothing to delete.")
            print(f"[INFO] Existing collections: {collection_names}")
            print(f"[INFO] No action taken - database unchanged.")
            
            # VERIFICATION: Double-check that we didn't accidentally create the collection
            # by listing collections again
            verify_collections = client.list_collections()
            verify_names = [col.name for col in verify_collections]
            if collection_name in verify_names:
                print(f"[WARNING] Collection '{collection_name}' was created unexpectedly!")
                print(f"[WARNING] This should not happen. Please report this issue.")
            else:
                print(f"[INFO] Verified: Collection '{collection_name}' still does not exist.")
            
            # Close client and return immediately - do NOT call get_collection() or any other method
            return True
        
        # Collection exists, get info before deletion
        # Use get_collection() NOT get_or_create_collection() to avoid creating new collections
        chunk_count = 0
        try:
            # Only get existing collection - this should not create anything
            collection = client.get_collection(name=collection_name)
            # Try to get count, but handle errors gracefully
            try:
                chunk_count = collection.count()
            except Exception as count_error:
                # If count() fails, try alternative method
                try:
                    all_data = collection.get()
                    ids = all_data.get('ids', [])
                    chunk_count = len(ids) if ids else 0
                except:
                    chunk_count = 0
                    print(f"[WARNING] Could not determine chunk count: {count_error}")
            
            print(f"[INFO] Collection '{collection_name}' contains {chunk_count:,} chunks")
        except Exception as e:
            # If get_collection fails, the collection might have been deleted between check and get
            error_msg = str(e).lower()
            if "does not exist" in error_msg or "not found" in error_msg:
                print(f"[INFO] Collection '{collection_name}' does not exist. Nothing to delete.")
                return True
            else:
                print(f"[WARNING] Could not get collection info: {e}")
                chunk_count = 0
        
        # Confirmation
        if not confirm:
            print()
            print("=" * 70)
            print("WARNING: This will permanently delete the collection!")
            print("=" * 70)
            print(f"Package: {package}")
            print(f"Collection: {collection_name}")
            print(f"Chunks: {chunk_count:,}")
            print()
            response = input("Are you sure you want to delete this collection? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("[INFO] Deletion cancelled.")
                return False
        
        # Delete the collection
        print()
        print(f"[INFO] Deleting collection '{collection_name}'...")
        try:
            client.delete_collection(name=collection_name)
            print(f"[SUCCESS] Collection '{collection_name}' deleted successfully!")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to delete collection: {e}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Failed to delete collection: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Delete a ChromaDB collection for a specific package",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Delete netket collection (with confirmation prompt)
  python delete_collection.py --package netket
  
  # Delete netket collection (skip confirmation)
  python delete_collection.py --package netket --confirm
        """
    )
    parser.add_argument(
        "--package",
        type=str,
        required=True,
        help="Package name to delete (e.g. netket, itensors, itensormps, qiskit, qiskit-nature, qiskit-algorithms, qiskit-aer)",
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
        "--confirm",
        action="store_true",
        default=False,
        help="Skip confirmation prompt (use with caution!)",
    )
    
    args = parser.parse_args()

    # Set project root
    if args.project_root:
        project_root = Path(args.project_root).resolve()
        set_project_root(project_root)
    else:
        try:
            # Get script file location
            script_file = Path(__file__).resolve()
            script_dir = script_file.parent  # MCP_servers/retrieve-mcp
            mcp_servers_dir = script_dir.parent  # MCP_servers
            project_root_candidate = mcp_servers_dir.parent  # project_root
            
            # Verify this is the correct project root
            authority_lib = project_root_candidate / "authority_library"
            if authority_lib.exists() and authority_lib.is_dir():
                detected_root = project_root_candidate
            else:
                # Search upward
                current = script_dir
                detected_root = None
                max_levels = 10
                level = 0
                while current != current.parent and level < max_levels:
                    authority_lib = current / "authority_library"
                    if authority_lib.exists() and authority_lib.is_dir():
                        detected_root = current
                        break
                    current = current.parent
                    level += 1
                
                if detected_root is None:
                    detected_root = find_project_root()
            
            project_root = detected_root
            set_project_root(project_root)
        except Exception as e:
            print(f"[ERROR] Failed to auto-detect project root: {e}")
            print("[ERROR] Please provide --project-root argument")
            sys.exit(1)

    print("=" * 70)
    print("Delete ChromaDB Collection")
    print("=" * 70)
    print()

    # Delete the collection
    success = delete_collection(
        package=args.package,
        chroma_path=args.chroma_path,
        project_root=project_root,
        confirm=args.confirm
    )

    print()
    print("=" * 70)
    if success:
        print("Operation completed successfully!")
    else:
        print("Operation failed or was cancelled.")
    print("=" * 70)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

