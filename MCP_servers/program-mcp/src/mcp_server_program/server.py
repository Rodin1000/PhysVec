"""MCP Program Server - FastMCP server implementation."""

import json
import logging
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

# Initialize logger
logger = logging.getLogger(__name__)

# Create a FastMCP server instance
mcp = FastMCP("Program Execution Service")

# Store configuration
_program_path: Optional[Path] = None
_default_timeout: int = 30


def set_program_path(file_path: str) -> None:
    """Set the program file path for code execution.
    
    Args:
        file_path: The program file path
    """
    global _program_path
    _program_path = Path(file_path).resolve()
    logger.info("Program file path set to: %s", _program_path)


def set_default_timeout(timeout: int) -> None:
    """Set the default timeout for code execution.
    
    Args:
        timeout: Default timeout in seconds
    """
    global _default_timeout
    _default_timeout = timeout
    logger.info("Default timeout set to: %d seconds", _default_timeout)


def _find_project_root(start_path: Path) -> Optional[Path]:
    """
    Find project root directory by looking for Project.toml or authority_library.
    
    Args:
        start_path: Starting path for search
        
    Returns:
        Path to project root directory, or None if not found
    """
    current = start_path.resolve()
    while current != current.parent:
        # Check for Project.toml (Julia project file)
        project_toml = current / "Project.toml"
        if project_toml.exists() and project_toml.is_file():
            return current
        # Also check for authority_library as fallback
        authority_lib = current / "authority_library"
        if authority_lib.exists() and authority_lib.is_dir():
            return current
        current = current.parent
    return None


def _find_venv_python() -> Optional[str]:
    """
    Find Python interpreter in virtual environment.
    Checks program file's directory and its parent directories for venv/ or .venv/.
    
    Returns:
        Path to Python interpreter in venv, or None if not found
    """
    if _program_path is None:
        return None
    
    # Get directory containing the program file
    program_dir = _program_path.parent
    
    # Check common virtual environment directory names
    venv_names = ["venv", ".venv"]
    
    # Check in program_dir and its parent directories (up to project root)
    current_dir = program_dir.resolve()
    max_levels = 5  # Limit search depth
    
    for _ in range(max_levels):
        for venv_name in venv_names:
            venv_dir = current_dir / venv_name
            
            if venv_dir.exists() and venv_dir.is_dir():
                # Determine Python executable path based on OS
                if platform.system() == "Windows":
                    python_exe = venv_dir / "Scripts" / "python.exe"
                else:
                    python_exe = venv_dir / "bin" / "python"
                
                if python_exe.exists() and python_exe.is_file():
                    logger.info("Found virtual environment: %s", venv_dir)
                    return str(python_exe)
        
        # Move up one directory level
        parent = current_dir.parent
        if parent == current_dir:  # Reached root
            break
        current_dir = parent
    
    return None


def _create_run_log_file(
    file_path: Path,
    file_type: str,
) -> Path:
    """
    Create initial run log JSON file before program execution.
    
    Args:
        file_path: Path to the program file to be executed
        file_type: File type suffix ("_py" or "_jl")
        
    Returns:
        Path to the created log file
    """
    try:
        # Get program file directory and name
        program_dir = file_path.parent
        program_name = file_path.stem  # filename without extension
        
        # Create log filename: run_log_xxx_py/jl.json
        log_filename = f"run_log_{program_name}{file_type}.json"
        log_path = program_dir / log_filename
        
        # Create initial log data (empty stdout/stderr, execution not started yet)
        initial_log_data = {
            "program_path": str(file_path.resolve()),
            "stdout": "",
            "stderr": "",
            "exitcode": None,
        }
        
        # Write initial JSON file
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(initial_log_data, f, indent=2, ensure_ascii=False)
        
        logger.info("Created run log file: %s", log_path)
        return log_path
        
    except Exception as e:
        logger.warning("Failed to create run log file: %s", str(e))
        # Return None or raise? For now, return a path even if creation failed
        # so the rest of the code can continue
        program_dir = file_path.parent
        program_name = file_path.stem
        log_filename = f"run_log_{program_name}{file_type}.json"
        return program_dir / log_filename


def _update_run_log(
    log_path: Path,
    stdout: str,
    stderr: str,
    exitcode: Optional[int] = None,
) -> None:
    """
    Update run log JSON file with execution results.
    
    Args:
        log_path: Path to the log file (created by _create_run_log_file)
        stdout: Standard output from execution
        stderr: Standard error from execution
        exitcode: Process exit code (0 = success, non-zero = error, None = error/timeout)
    """
    try:
        # Read existing log data if file exists
        if log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    log_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                # If file is corrupted or can't be read, create new data
                log_data = {}
        else:
            log_data = {}
        
        # Update log data with execution results
        log_data["stdout"] = stdout
        log_data["stderr"] = stderr
        log_data["exitcode"] = exitcode
        
        # Write updated JSON file
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        logger.info("Updated run log file: %s", log_path)
        
    except Exception as e:
        logger.warning("Failed to update run log file: %s", str(e))


@mcp.tool()
def run_julia_file(
    file_path: str,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute a Julia file and return the execution results.
    
    This tool runs a Julia file and returns stdout, stderr, and exit code
    for LLM evaluation of code execution.
    
    Args:
        file_path: Path to Julia file (absolute or relative to program file's directory)
        timeout: Execution timeout in seconds (default: server's default_timeout)
        
    Returns:
        Dictionary containing:
        - exitcode: Process exit code (0 = success, non-zero = error, None = error/timeout)
        - stdout: Standard output (captured and returned to LLM for evaluation)
        - stderr: Standard error (captured and returned to LLM for evaluation)
        - timeout: True if execution timed out, False otherwise
    """
    if _program_path is None:
        return {
            "exitcode": None,
            "stdout": "",
            "stderr": "Program file path not set",
            "timeout": False,
        }
    
    # Use default timeout if not provided
    exec_timeout = timeout if timeout is not None else _default_timeout
    
    try:
        # Resolve file path (relative to program file's directory if not absolute)
        program_dir = _program_path.parent
        if Path(file_path).is_absolute():
            validated_path = Path(file_path).resolve()
        else:
            validated_path = (program_dir / file_path).resolve()
        
        if not validated_path.exists():
            return {
                "exitcode": None,
                "stdout": "",
                "stderr": f"File not found: {file_path}",
                "timeout": False,
            }
        
        # Find project root directory
        project_root = _find_project_root(validated_path) or _find_project_root(_program_path.parent)
        if project_root is None:
            project_root = program_dir
            logger.warning("Could not find project root, using program directory: %s", program_dir)
        else:
            logger.info("Found project root: %s", project_root)
        
        # Calculate relative path from project root (for consistency with run_program.py)
        if validated_path.is_absolute() and project_root:
            try:
                relative_path = validated_path.relative_to(project_root)
                file_arg = str(relative_path)
            except ValueError:
                # File is outside project root, use absolute path
                file_arg = str(validated_path)
        else:
            file_arg = str(validated_path)
        
        # Execute Julia file
        cmd = [
            "julia",
            "--color=no",
            "--startup-file=no",
            "--project=@.",
            file_arg,
        ]
        
        logger.info("Executing Julia file: %s (timeout: %ds, cwd: %s)", file_arg, exec_timeout, project_root)
        
        # Create log file before execution
        log_path = _create_run_log_file(validated_path, "_jl")
        
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=exec_timeout,
            cwd=str(project_root),  # Use project root as working directory
            stdin=subprocess.DEVNULL,  # Explicitly close stdin to prevent subprocess from waiting for input
        )
        
        # Update log file with execution results
        _update_run_log(log_path, proc.stdout, proc.stderr, proc.returncode)
        
        return {
            "exitcode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timeout": False,
        }
        
    except subprocess.TimeoutExpired:
        logger.warning("Julia execution timeout: %s (timeout: %ds)", file_path, exec_timeout)
        # Update log file with timeout error
        if 'log_path' in locals():
            _update_run_log(log_path, "", f"Execution timeout after {exec_timeout}s", None)
        return {
            "exitcode": None,
            "stdout": "",
            "stderr": f"Execution timeout after {exec_timeout}s",
            "timeout": True,
        }
    except FileNotFoundError:
        logger.error("Julia executable not found in PATH")
        return {
            "exitcode": None,
            "stdout": "",
            "stderr": "julia executable not found in PATH",
            "timeout": False,
        }
    except ValueError as e:
        logger.error("Path validation error: %s", str(e))
        return {
            "exitcode": None,
            "stdout": "",
            "stderr": str(e),
            "timeout": False,
        }
    except Exception as e:
        logger.error("Unexpected error executing Julia file: %s", str(e))
        return {
            "exitcode": None,
            "stdout": "",
            "stderr": f"Unexpected error: {str(e)}",
            "timeout": False,
        }


@mcp.tool()
def run_python_file(
    file_path: str,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute a Python file and return the execution results.
    
    This tool runs a Python file and returns stdout, stderr, and exit code
    for LLM evaluation of code execution. Uses virtual environment if found.
    
    Args:
        file_path: Path to Python file (absolute or relative to program file's directory)
        timeout: Execution timeout in seconds (default: server's default_timeout)
        
    Returns:
        Dictionary containing:
        - exitcode: Process exit code (0 = success, non-zero = error, None = error/timeout)
        - stdout: Standard output (captured and returned to LLM for evaluation)
        - stderr: Standard error (captured and returned to LLM for evaluation)
        - timeout: True if execution timed out, False otherwise
    """
    if _program_path is None:
        return {
            "exitcode": None,
            "stdout": "",
            "stderr": "Program file path not set",
            "timeout": False,
        }
    
    # Use default timeout if not provided
    exec_timeout = timeout if timeout is not None else _default_timeout
    
    try:
        # Resolve file path (relative to program file's directory if not absolute)
        program_dir = _program_path.parent
        if Path(file_path).is_absolute():
            validated_path = Path(file_path).resolve()
        else:
            validated_path = (program_dir / file_path).resolve()
        
        if not validated_path.exists():
            return {
                "exitcode": None,
                "stdout": "",
                "stderr": f"File not found: {file_path}",
                "timeout": False,
            }
        
        # Find project root directory
        project_root = _find_project_root(validated_path) or _find_project_root(_program_path.parent)
        if project_root is None:
            project_root = program_dir
            logger.warning("Could not find project root, using program directory: %s", program_dir)
        else:
            logger.info("Found project root: %s", project_root)
        
        # Find virtual environment Python interpreter
        python_exe = _find_venv_python()
        if python_exe is None:
            python_exe = "python"  # Fallback to system Python
            logger.info("No virtual environment found, using system Python")
        else:
            logger.info("Using virtual environment Python: %s", python_exe)
        
        # Calculate relative path from project root (for consistency with run_program.py)
        if validated_path.is_absolute() and project_root:
            try:
                relative_path = validated_path.relative_to(project_root)
                file_arg = str(relative_path)
            except ValueError:
                # File is outside project root, use absolute path
                file_arg = str(validated_path)
        else:
            file_arg = str(validated_path)
        
        # Execute Python file
        cmd = [python_exe, file_arg]
        
        logger.info("Executing Python file: %s (timeout: %ds, cwd: %s)", file_arg, exec_timeout, project_root)
        
        # Create log file before execution
        log_path = _create_run_log_file(validated_path, "_py")
        
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=exec_timeout,
            cwd=str(project_root),  # Use project root as working directory
            stdin=subprocess.DEVNULL,  # Explicitly close stdin to prevent subprocess from waiting for input
        )
        
        # Update log file with execution results
        _update_run_log(log_path, proc.stdout, proc.stderr, proc.returncode)
        
        return {
            "exitcode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timeout": False,
        }
        
    except subprocess.TimeoutExpired:
        logger.warning("Python execution timeout: %s (timeout: %ds)", file_path, exec_timeout)
        # Update log file with timeout error
        if 'log_path' in locals():
            _update_run_log(log_path, "", f"Execution timeout after {exec_timeout}s", None)
        return {
            "exitcode": None,
            "stdout": "",
            "stderr": f"Execution timeout after {exec_timeout}s",
            "timeout": True,
        }
    except FileNotFoundError:
        logger.error("Python executable not found in PATH")
        return {
            "exitcode": None,
            "stdout": "",
            "stderr": "python executable not found in PATH",
            "timeout": False,
        }
    except ValueError as e:
        logger.error("Path validation error: %s", str(e))
        return {
            "exitcode": None,
            "stdout": "",
            "stderr": str(e),
            "timeout": False,
        }
    except Exception as e:
        logger.error("Unexpected error executing Python file: %s", str(e))
        return {
            "exitcode": None,
            "stdout": "",
            "stderr": f"Unexpected error: {str(e)}",
            "timeout": False,
        }


def run_server(program_path: str, default_timeout: int = 30) -> None:
    """
    Run the MCP server.
    
    Args:
        program_path: Path to the program file
        default_timeout: Default timeout in seconds
    """
    set_program_path(program_path)
    set_default_timeout(default_timeout)
    
    # Run the FastMCP server
    mcp.run()

