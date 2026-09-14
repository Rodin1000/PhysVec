# MCP ORCA DFT Server

Minimal MCP server for executing ORCA quantum chemistry calculations.

## Installation

```bash
cd MCP_servers/program-dft-mcp
pip install -e .
```

## Usage

```bash
mcp-server-program-dft --input-path /path/to/input.inp --timeout 600
```

## Tool

### `run_orca`

Execute ORCA input file.

**Parameters:**
- `file_path`: Path to ORCA input file (relative to input-path directory or absolute)
- `timeout`: Execution timeout in seconds (optional)

**Returns:**
- `exitcode`: Process exit code
- `stdout`: ORCA output
- `stderr`: Error messages
- `timeout`: True if timed out

## Environment

Set `ORCA_EXECUTABLE` environment variable to specify ORCA path (default: `/home/xiangfei/orca_install_dir/orca`).
