"""Entry point for ORCA MCP server."""

import argparse
import sys
from pathlib import Path
from mcp_server_program_dft.server import run_server


def main():
    parser = argparse.ArgumentParser(description="ORCA DFT MCP Server")
    parser.add_argument("--input-path", required=True, help="Path to ORCA input file")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds")
    args = parser.parse_args()
    
    inp_path = Path(args.input_path)
    if not inp_path.exists():
        print(f"Error: Input file not found: {inp_path}", file=sys.stderr)
        sys.exit(1)
    
    run_server(str(inp_path.resolve()), args.timeout)


if __name__ == "__main__":
    main()
