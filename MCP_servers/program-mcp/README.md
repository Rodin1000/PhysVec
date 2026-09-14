# MCP Program Server

A Model Context Protocol (MCP) server that provides code execution capabilities for Python and Julia files.

## Overview

This MCP server enables AI assistants to execute Python and Julia code files. The server is configured with a specific program file path, and code execution uses the program file's directory as the working directory.

## Features

- Execute Python files using the `python` command (with automatic virtual environment detection)
- Execute Julia files using the `julia` command
- Automatic virtual environment detection for Python execution
- Configurable timeout for code execution
- Returns stdout, stderr, and exit code for LLM evaluation

## Installation

```bash
cd MCP_servers/program-mcp
pip install -e .
```

Or install dependencies only:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line

```bash
mcp-server-program --program-path /path/to/program.py --timeout 30 --log-level INFO
```

### Arguments

- `--program-path` (required): Path to the program file
- `--timeout` (optional, default: 30): Default timeout in seconds
- `--log-level` (optional, default: INFO): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

### MCP Tools

#### `run_python_file`

Execute a Python file and return execution results.

**Parameters:**
- `file_path` (str): Path to Python file (absolute or relative to program file's directory)
- `timeout` (int, optional): Execution timeout in seconds (default: server's default_timeout)

**Returns:**
```python
{
    "exitcode": int,        # Process exit code (0 = success, non-zero = error, None = error/timeout)
    "stdout": str,          # Standard output (captured and returned to LLM for evaluation)
    "stderr": str,          # Standard error (captured and returned to LLM for evaluation)
    "timeout": bool         # True if execution timed out, False otherwise
}
```

#### `run_julia_file`

Execute a Julia file and return execution results.

**Parameters:**
- `file_path` (str): Path to Julia file (absolute or relative to program file's directory)
- `timeout` (int, optional): Execution timeout in seconds (default: server's default_timeout)

**Returns:**
```python
{
    "exitcode": int,        # Process exit code (0 = success, non-zero = error, None = error/timeout)
    "stdout": str,          # Standard output (captured and returned to LLM for evaluation)
    "stderr": str,          # Standard error (captured and returned to LLM for evaluation)
    "timeout": bool         # True if execution timed out, False otherwise
}
```

## Virtual Environment Support

- **Automatic Detection**: The server automatically detects virtual environments (`venv/` or `.venv/`) in the program file's directory or its parent directories
- **Python Execution**: When executing Python files, the server uses the virtual environment's Python interpreter if found, otherwise falls back to system Python

## Integration with MCP_toolbox.py

The server can be integrated into `send_chat_through_mcp_dynamic`:

```python
from utils.MCP_toolbox import send_chat_through_mcp_dynamic

response = await send_chat_through_mcp_dynamic(
    user_model="deepseek/deepseek-chat-v3.1:free",
    user_prompt="Execute test.py and show me the output",
    init_servers=["program-mcp"],
    program_path="/path/to/program.py",
    program_timeout=60,
)
```

## Requirements

- Python >= 3.11
- `mcp>=1.3.0` (with server and CLI extras)
- `python` executable in PATH (for Python execution)
- `julia` executable in PATH (for Julia execution)

## License

MIT

