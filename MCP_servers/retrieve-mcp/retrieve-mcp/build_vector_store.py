"""Build vector store for RAG retrieval - Entry Point.

This is the user-facing entry point script for building vector stores.
Users can edit and run this script directly.

Usage:
    python build_vector_store.py --package itensormps
"""

import argparse
import sys
from pathlib import Path

# Import build functions from the module
sys.path.insert(0, str(Path(__file__).parent / "src"))
from mcp_server_retrieve.vector_store_builder import build_vector_store
from mcp_server_retrieve.retrieval import find_project_root, set_project_root


def main():
    """Main entry point - Edit this section to customize behavior."""
    parser = argparse.ArgumentParser(
        description="Build vector store for RAG retrieval from authority_library packages"
    )
    parser.add_argument(
        "--package",
        type=str,
        required=True,
        help="Package name to build (e.g. itensormps, itensors, netket, qiskit, qiskit-nature, qiskit-algorithms, qiskit-aer)",
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
    
    args = parser.parse_args()

    # Set project root - use same logic as check_vector_store.py for consistency
    if args.project_root:
        project_root = Path(args.project_root).resolve()
        set_project_root(project_root)
    else:
        try:
            # Get script file location
            script_file = Path(__file__).resolve()
            # Script is at: project_root/MCP_servers/retrieve-mcp/build_vector_store.py
            script_dir = script_file.parent  # MCP_servers/retrieve-mcp
            mcp_servers_dir = script_dir.parent  # MCP_servers
            project_root_candidate = mcp_servers_dir.parent  # project_root
            
            # Verify this is the correct project root by checking for authority_library
            authority_lib = project_root_candidate / "authority_library"
            if authority_lib.exists() and authority_lib.is_dir():
                project_root = project_root_candidate
            else:
                # If calculated path doesn't work, search upward from script directory
                current = script_dir
                project_root = None
                max_levels = 10
                level = 0
                while current != current.parent and level < max_levels:
                    authority_lib = current / "authority_library"
                    if authority_lib.exists() and authority_lib.is_dir():
                        project_root = current
                        break
                    current = current.parent
                    level += 1
                
                if project_root is None:
                    # Last resort: try find_project_root()
                    project_root = find_project_root()
            
            # Verify authority_library exists
            authority_lib = project_root / "authority_library"
            if not authority_lib.exists() or not authority_lib.is_dir():
                raise FileNotFoundError(
                    f"Could not find authority_library directory.\n"
                    f"  Script location: {script_file}\n"
                    f"  Detected root: {project_root}\n"
                    f"  Current working directory: {Path.cwd()}"
                )
            
            set_project_root(project_root)
        except Exception as e:
            print(f"[ERROR] Failed to auto-detect project root: {e}")
            print(f"[ERROR] Script location: {Path(__file__).resolve()}")
            print(f"[ERROR] Current working directory: {Path.cwd()}")
            print("[ERROR] Please provide --project-root argument explicitly")
            sys.exit(1)

    print("=" * 60)
    print("Building vector store for package:", args.package)
    print("=" * 60)
    
    try:
        build_vector_store(
            package=args.package,
            chroma_path=args.chroma_path,
            project_root=project_root,
        )
        print("\n" + "=" * 60)
        print("Vector store built successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\n[ERROR] Failed to build vector store: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
