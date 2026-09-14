"""Main entry point for the MCP Retrieve Server."""

import argparse
import logging
import sys
from pathlib import Path

# Import server
from mcp_server_retrieve.server import run_server
from mcp_server_retrieve.retrieval import DEFAULT_PACKAGE, get_package_configs, set_project_root, find_project_root


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="MCP Retrieve Server - RAG retrieval service")
    parser.add_argument(
        "--chroma-path",
        type=str,
        default="authority_library/rag_store/rag_chroma",
        help="Path to ChromaDB storage directory (default: authority_library/rag_store/rag_chroma, relative to project root)",
    )
    parser.add_argument(
        "--package",
        type=str,
        default=DEFAULT_PACKAGE,
        help=f"Default package name (default: {DEFAULT_PACKAGE})",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Path to project root directory (auto-detected if not provided)",
    )
    parser.add_argument(
        "--rt-target-dir",
        type=str,
        default=None,
        help="Target directory for saving retrieval results (optional). If set, retrieval results will be automatically saved to files instead of being returned to LLM.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level (default: INFO)",
    )
    return parser.parse_args()


def setup_logging(log_level: str) -> None:
    """Configure logging.
    
    Args:
        log_level: Logging level
    """
    import sys
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,  # Output to stderr to avoid polluting stdout (JSONRPC communication channel)
    )


def main() -> None:
    """
    Main entry point for the MCP Retrieve Server.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Set project root first (needed for resolving relative paths)
    project_root = None
    if args.project_root:
        project_root = Path(args.project_root)
        if not project_root.is_absolute():
            project_root = Path.cwd() / project_root
        project_root = project_root.resolve()
        try:
            set_project_root(project_root)
        except Exception as e:
            logger.error("Failed to set project root: %s", str(e))
            sys.exit(1)
    else:
        try:
            detected_root = find_project_root()
            set_project_root(detected_root)
            project_root = detected_root
        except Exception as e:
            logger.error("Failed to auto-detect project root: %s", str(e))
            logger.error("Please provide --project-root argument")
            sys.exit(1)
    
    # Validate chroma path (if it's a relative path, resolve relative to project root)
    chroma_path = Path(args.chroma_path)
    if not chroma_path.is_absolute():
        # Resolve relative to project root (not current working directory)
        chroma_path = project_root / chroma_path
    chroma_path = str(chroma_path.resolve())
    
    # Validate package name
    configs = get_package_configs()
    if args.package not in configs:
        logger.error("Invalid package name: %s. Available packages: %s", args.package, list(configs.keys()))
        sys.exit(1)
    
    logger.info("Starting MCP Retrieve Server")
    logger.info("Project root: %s", project_root)
    logger.info("ChromaDB path: %s", chroma_path)
    logger.info("Default package: %s", args.package)
    if args.rt_target_dir:
        logger.info("Retrieval target directory: %s", args.rt_target_dir)
    logger.info("Log level: %s", args.log_level)
    
    # Resolve rt_target_dir if provided
    rt_target_dir = None
    if args.rt_target_dir:
        rt_target_dir_path = Path(args.rt_target_dir)
        if not rt_target_dir_path.is_absolute():
            rt_target_dir_path = Path.cwd() / rt_target_dir_path
        rt_target_dir = str(rt_target_dir_path.resolve())
    
    # Run the server
    try:
        run_server(chroma_path=chroma_path, default_package=args.package, project_root=str(project_root), rt_target_dir=rt_target_dir)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error("Server error: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()

