# This program provides functions to execute programs and collect output information
import subprocess
import re
import os
import platform
from pathlib import Path

def _find_venv_python(target_file: str, project_root: str) -> str:
    """
    Find Python interpreter in virtual environment.
    Checks target file's directory and its parent directories (up to 10 levels) for venv/ or .venv/.
    This matches the behavior of program-mcp's _find_venv_python function.
    
    Args:
        target_file: Path to the Python file to execute
        project_root: Project root directory (not used for limiting search, but for resolving relative paths)
        
    Returns:
        Path to Python interpreter in venv, or "python" if not found
    """
    target_path = Path(target_file).resolve()
    if not target_path.is_absolute():
        target_path = (Path(project_root) / target_file).resolve()
    
    # Get directory containing the target file
    current_dir = target_path.parent.resolve()
    
    # Check common virtual environment directory names
    venv_names = ["venv", ".venv"]
    
    # Check in target file's directory and its parent directories (up to 10 levels)
    # This matches program-mcp's behavior: search up to 10 levels without restricting to project_root
    max_levels = 10
    
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
                    return str(python_exe)
        
        # Move up one directory level
        parent = current_dir.parent
        if parent == current_dir:  # Reached root
            break
        current_dir = parent
    
    return "python"  # Fallback to system Python


def run_julia_file(target_file: str, project_root: str = ".", timeout: int = 30) -> dict:
    """
    Execute a Julia file and return the execution results.
    
    Args:
        target_file: Path to Julia file (relative to project_root or absolute)
        project_root: Project root directory (default: ".")
        timeout: Execution timeout in seconds (default: 30)
        
    Returns:
        Dictionary containing:
        - exitcode: Process exit code (0 = success, non-zero = error, None = error/timeout)
        - stdout: Standard output
        - stderr: Standard error
        - timeout: True if execution timed out, False otherwise
    """
    # Resolve target file path
    target_path = Path(target_file).resolve()
    project_root_path = Path(project_root).resolve()
    
    # If target_file is within project_root, use relative path
    # Otherwise, use absolute path
    try:
        # Try to get relative path if target is within project_root
        file_arg = str(target_path.relative_to(project_root_path))
    except ValueError:
        # Target is outside project_root, use absolute path
        file_arg = str(target_path)
    
    cmd = ["julia", "--color=no", "--startup-file=no", "--project=@.", file_arg]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=timeout,
            cwd=os.path.abspath(project_root)  # execute under the root path of this project
        )
        return {
            "exitcode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timeout": False
        }
    except subprocess.TimeoutExpired as e:
        return {
            "exitcode": None,
            "stdout": "",
            "stderr": f"Execution timeout after {timeout}s",
            "timeout": True
        }
    except FileNotFoundError:
        return {
            "exitcode": None,
            "stdout": "",
            "stderr": "julia executable not found in PATH",
            "timeout": False
        }


def run_python_file(target_file: str, project_root: str = ".", timeout: int = 30) -> dict:
    """
    Execute a Python file and return the execution results.
    
    This function runs a Python file and returns stdout, stderr, and exit code.
    Uses virtual environment if found in the target file's directory or its parent directories.
    
    Args:
        target_file: Path to Python file (relative to project_root or absolute)
        project_root: Project root directory (default: ".")
        timeout: Execution timeout in seconds (default: 30)
        
    Returns:
        Dictionary containing:
        - exitcode: Process exit code (0 = success, non-zero = error, None = error/timeout)
        - stdout: Standard output
        - stderr: Standard error
        - timeout: True if execution timed out, False otherwise
    """
    # Find virtual environment Python interpreter
    python_exe = _find_venv_python(target_file, project_root)
    
    # Resolve target file path
    target_path = Path(target_file).resolve()
    project_root_path = Path(project_root).resolve()
    
    # If target_file is within project_root, use relative path
    # Otherwise, use absolute path
    try:
        # Try to get relative path if target is within project_root
        file_arg = str(target_path.relative_to(project_root_path))
    except ValueError:
        # Target is outside project_root, use absolute path
        file_arg = str(target_path)
    
    cmd = [python_exe, file_arg]
    
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=timeout,
            cwd=os.path.abspath(project_root),  # execute under the root path of this project
            stdin=subprocess.DEVNULL,  # Explicitly close stdin to prevent subprocess from waiting for input
        )
        return {
            "exitcode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timeout": False
        }
    except subprocess.TimeoutExpired as e:
        return {
            "exitcode": None,
            "stdout": "",
            "stderr": f"Execution timeout after {timeout}s",
            "timeout": True
        }
    except FileNotFoundError:
        return {
            "exitcode": None,
            "stdout": "",
            "stderr": f"Python executable not found: {python_exe}",
            "timeout": False
        }