"""Main entry point for the MCP Program Server."""

import argparse
import logging
import sys
from pathlib import Path

from mcp_server_program.server import run_server


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="MCP Program Server - Code execution service")
    parser.add_argument(
        "--program-path",
        type=str,
        required=True,
        help="Path to the program file to execute (required)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Default timeout in seconds for code execution (default: 30)",
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
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    """
    Main entry point for the MCP Program Server.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Validate program file
    program_path = Path(args.program_path)
    if not program_path.exists():
        logger.error("Program file does not exist: %s", program_path)
        sys.exit(1)
    if not program_path.is_file():
        logger.error("Program path is not a file: %s", program_path)
        sys.exit(1)
    
    # Convert to absolute path
    program_path = program_path.resolve()
    
    logger.info("Starting MCP Program Server")
    logger.info("Program file: %s", program_path)
    logger.info("Default timeout: %d seconds", args.timeout)
    logger.info("Log level: %s", args.log_level)
    
    # Run the server
    try:
        run_server(program_path=str(program_path), default_timeout=args.timeout)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error("Server error: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()

