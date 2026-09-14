"""Minimal MCP server for ORCA execution."""

import os
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ORCA DFT Server")

_input_path: Optional[Path] = None
_timeout: int = 600


def set_input_path(path: str) -> None:
    global _input_path
    _input_path = Path(path).resolve()


def set_timeout(timeout: int) -> None:
    global _timeout
    _timeout = timeout


@mcp.tool()
def run_orca(file_path: str, timeout: Optional[int] = None) -> Dict[str, Any]:
    """
    Execute an ORCA input file and return the execution results.
    
    ORCA execution format: orca <input_file.inp> > <output_file.out> 2>&1
    ORCA writes output to a .out file (same name as input but .out extension).
    
    Args:
        file_path: Path to ORCA .inp file (relative to input_path base or absolute)
        timeout: Execution timeout in seconds (default: from server config, typically 600 for DFT calculations)
        
    Returns:
        Dictionary containing:
        - exitcode: Process exit code (0 = success, non-zero = error, None = error/timeout)
        - stdout: Last 100 lines of ORCA output (most recent output for debugging)
        - stderr: Last 100 lines of error messages (most recent errors, extracted from output)
        - timeout: True if execution timed out, False otherwise
        - output_truncated: True if output was truncated (more than 100 lines), False otherwise
        - running_successfully: (if timeout) True if ORCA was running successfully when timed out
    """
    if _input_path is None:
        return {
            "exitcode": None,
            "stdout": "",
            "stderr": "Input path not set. Server not properly initialized.",
            "timeout": False
        }
    
    exec_timeout = timeout or _timeout
    orca_exe = os.environ.get("ORCA_EXECUTABLE", "/home/xiangfei/orca_install_dir/orca")
    
    # Resolve file path
    # If file_path is absolute, use it directly
    # Otherwise, resolve relative to _input_path's parent directory
    if Path(file_path).is_absolute():
        inp_file = Path(file_path).resolve()
    else:
        inp_file = (_input_path.parent / file_path).resolve()
    
    # Validate input file exists
    if not inp_file.exists():
        return {
            "exitcode": None,
            "stdout": "",
            "stderr": f"ORCA input file not found: {file_path} (resolved to: {inp_file})",
            "timeout": False
        }
    
    # ORCA output file (same name as input but .out extension)
    output_file = inp_file.with_suffix('.out')
    
    # Clean up ORCA output files from previous runs (except .inp file)
    # ORCA creates many files: .out, .gbw, .xyz, .trj.xyz, .densities, .densitiesinfo, 
    # .property.txt, .bibtex, .cpcm, .cpcm_corr, etc.
    orca_output_extensions = [
        '.out', '.gbw', '.xyz', '.trj.xyz', '.densities', '.densitiesinfo',
        '.property.txt', '.bibtex', '.cpcm', '.cpcm_corr', '.hess', '.opt',
        '.tmp', '.engrad', '.molden', '.cube', '.wfn', '.wfx', '.molden.input'
    ]
    
    input_dir = inp_file.parent
    input_stem = inp_file.stem
    
    # Remove all ORCA output files with the same stem
    cleaned_files = []
    for ext in orca_output_extensions:
        output_file_path = input_dir / f"{input_stem}{ext}"
        if output_file_path.exists() and output_file_path != inp_file:
            try:
                output_file_path.unlink()
                cleaned_files.append(output_file_path.name)
            except Exception as e:
                # Log but don't fail if cleanup fails
                pass
    
    # Also clean up any files matching ORCA patterns in the directory
    # (e.g., files starting with input_stem but with different extensions)
    # But preserve log files, query files, and refinement history files
    protected_patterns = ['run_log_', 'report_', 'iteration_log_', 'refinement_history_', 'optimized_']
    protected_dirs = ['query', 'logs']  # Don't touch files in these directories
    
    if input_dir.exists():
        for file_path in input_dir.iterdir():
            if file_path.is_file() and file_path.stem == input_stem and file_path.suffix != '.inp':
                # Skip if it's a protected file (log, report, etc.)
                if any(file_path.name.startswith(pattern) for pattern in protected_patterns):
                    continue
                # Skip if it's in a protected directory
                if any(protected_dir in str(file_path.parent) for protected_dir in protected_dirs):
                    continue
                # Skip if it's already been cleaned
                if file_path.name in cleaned_files:
                    continue
                try:
                    file_path.unlink()
                    cleaned_files.append(file_path.name)
                except Exception:
                    pass
    
    # ORCA command: orca input.inp
    # ORCA automatically writes to input.out (same name, .out extension)
    cmd = [orca_exe, str(inp_file)]
    
    try:
        # Execute ORCA
        # ORCA writes output to .out file, but also may write to stderr
        proc = subprocess.run(
            cmd,
            capture_output=True,  # Capture both stdout and stderr
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=exec_timeout,
            cwd=str(inp_file.parent),
            stdin=subprocess.DEVNULL,  # Explicitly close stdin
        )
        
        # Read ORCA output file if it exists
        output_content = ""
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8', errors='replace') as f:
                    output_content = f.read()
            except Exception as e:
                output_content = f"Error reading ORCA output file: {str(e)}"
        
        # Combine stdout and stderr (ORCA may write to both)
        combined_output = proc.stdout + proc.stderr
        if output_content:
            # Prefer output file content if available
            full_output = output_content
            if combined_output:
                full_output = output_content + "\n\n--- Process output ---\n" + combined_output
        else:
            full_output = combined_output
        
        # Determine exitcode and error status
        exitcode = proc.returncode
        
        # Extract error messages from output (for LLM to see specific errors)
        # ORCA typically prints "ORCA TERMINATED NORMALLY" on success
        # and "ORCA aborted" or "ERROR" on failure
        error_messages = []
        stderr_content = ""
        
        # Check for ORCA success/failure indicators
        if exitcode != 0:
            # Process returned non-zero exit code - definitely an error
            error_messages.append(f"ORCA exited with code {exitcode}")
            stderr_content = full_output  # Include full output in stderr for error context
        elif "ORCA aborted" in full_output.upper():
            # ORCA aborted even if exitcode is 0
            exitcode = 1
            error_messages.append("ORCA aborted during execution")
            stderr_content = full_output
        elif "ORCA TERMINATED NORMALLY" not in full_output.upper() and full_output:
            # No success message - might be an error
            if any(keyword in full_output.upper() for keyword in ["FATAL", "ABORT", "CRITICAL ERROR", "ERROR"]):
                exitcode = 1
                error_messages.append("Error detected in ORCA output (no 'ORCA TERMINATED NORMALLY' message)")
                stderr_content = full_output
        
        # Extract specific error lines for clearer error reporting
        if stderr_content:
            # Try to extract key error lines (lines containing ERROR, FATAL, ABORT, etc.)
            error_lines = []
            for line in full_output.split('\n'):
                line_upper = line.upper()
                if any(keyword in line_upper for keyword in ["ERROR", "FATAL", "ABORT", "CRITICAL", "FAILED", "EXCEPTION"]):
                    error_lines.append(line.strip())
            
            if error_lines:
                # Add extracted error lines to stderr for quick reference
                stderr_content = "\n".join(error_lines) + "\n\n--- Full ORCA Output ---\n" + full_output
        
        # Truncate output to last N lines for LLM to avoid slow responses and token limits
        # Use last 100 lines: most recent output/errors are most relevant for debugging
        MAX_OUTPUT_LINES = 100  # Take last 100 lines of output
        
        def truncate_to_last_lines(text: str, max_lines: int = MAX_OUTPUT_LINES) -> str:
            """Take the last N lines of output (most recent output/errors)."""
            if not text:
                return text
            lines = text.split('\n')
            if len(lines) <= max_lines:
                return text
            # Take last max_lines lines
            last_lines = lines[-max_lines:]
            truncated = '\n'.join(last_lines)
            truncated = f"... (truncated: showing last {max_lines} of {len(lines)} lines) ...\n\n" + truncated
            return truncated
        
        # Truncate stdout to last 100 lines
        truncated_stdout = truncate_to_last_lines(full_output, max_lines=MAX_OUTPUT_LINES)
        
        # Truncate stderr to last 100 lines (error messages are most relevant at the end)
        truncated_stderr = truncate_to_last_lines(stderr_content, max_lines=MAX_OUTPUT_LINES)
        
        # Return last 100 lines of output (stdout) and error messages (stderr)
        # LLM can access:
        # - stdout: Last 100 lines of ORCA output (most recent output for validation/debugging)
        # - stderr: Last 100 lines of error messages (most recent errors for analysis)
        # - exitcode: Success (0) or failure (non-zero)
        return {
            "exitcode": exitcode,
            "stdout": truncated_stdout,  # Last 100 lines of ORCA output for LLM to analyze
            "stderr": truncated_stderr,  # Last 100 lines of error messages if errors, empty if success
            "timeout": False,
            "output_truncated": len(full_output.split('\n')) > MAX_OUTPUT_LINES or (stderr_content and len(stderr_content.split('\n')) > MAX_OUTPUT_LINES)
        }
        
    except subprocess.TimeoutExpired:
        # Timeout occurred, but check if ORCA started successfully (no parser errors)
        # If ORCA was running without errors, consider it a success
        output_content = ""
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8', errors='replace') as f:
                    output_content = f.read()
            except Exception:
                pass
        
        # Check if ORCA started successfully (no parser/syntax errors)
        # Success indicators: ORCA banner appears, no fatal errors in output
        is_running_successfully = False
        if output_content:
            output_upper = output_content.upper()
            # Check for ORCA banner/startup (indicates it started)
            has_orca_banner = "ORCA" in output_upper and ("PROGRAM" in output_upper or "VERSION" in output_upper)
            # Check for fatal parser errors
            has_fatal_errors = any(keyword in output_upper for keyword in [
                "FATAL ERROR", "INPUT ERROR", "SYNTAX ERROR", "PARSER ERROR",
                "CANNOT READ INPUT", "INVALID KEYWORD", "UNKNOWN KEYWORD"
            ])
            # If ORCA started and no fatal errors, it's running successfully
            if has_orca_banner and not has_fatal_errors:
                is_running_successfully = True
        
        # Helper function to get last N lines (reuse from main logic)
        def truncate_to_last_lines(text: str, max_lines: int = 100) -> str:
            """Take the last N lines of output (most recent output/errors)."""
            if not text:
                return text
            lines = text.split('\n')
            if len(lines) <= max_lines:
                return text
            last_lines = lines[-max_lines:]
            truncated = '\n'.join(last_lines)
            truncated = f"... (truncated: showing last {max_lines} of {len(lines)} lines) ...\n\n" + truncated
            return truncated
        
        if is_running_successfully:
            # ORCA is running successfully (just taking longer than timeout)
            # This is considered success for input validation purposes
            # Use last 100 lines to see most recent progress
            truncated_output = truncate_to_last_lines(output_content, max_lines=100)
            
            return {
                "exitcode": 0,  # Treat as success
                "stdout": truncated_output + f"\n\n--- Note: ORCA execution timed out after {exec_timeout}s but was running successfully. Input file is valid. ---",
                "stderr": "",  # No errors
                "timeout": True,
                "running_successfully": True,  # Flag to indicate it was running when timed out
                "output_truncated": len(output_content.split('\n')) > 100
            }
        else:
            # Timeout but couldn't determine if it was running successfully
            # Use last 100 lines to see most recent errors/output
            truncated_output = truncate_to_last_lines(output_content, max_lines=100) if output_content else ""
            return {
                "exitcode": None,
                "stdout": truncated_output,
                "stderr": f"ORCA execution timeout after {exec_timeout}s. Could not determine if input is valid.\n\n--- Last 100 lines of output ---\n{truncated_output}",
                "timeout": True,
                "running_successfully": False
            }
    except FileNotFoundError:
        return {
            "exitcode": None,
            "stdout": "",
            "stderr": f"ORCA executable not found: {orca_exe}. Set ORCA_EXECUTABLE environment variable or ensure ORCA is in PATH.",
            "timeout": False
        }
    except Exception as e:
        return {
            "exitcode": None,
            "stdout": "",
            "stderr": f"Error executing ORCA: {str(e)}",
            "timeout": False
        }


def run_server(input_path: str, timeout: int = 600) -> None:
    set_input_path(input_path)
    set_timeout(timeout)
    mcp.run()
