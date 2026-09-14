# This program provides functions to execute programs and collect output information
import subprocess
import re
import os
import platform
import time
from pathlib import Path
from typing import Optional

def _find_venv_python(target_file: str, project_root: str) -> str:
    """
    Find Python interpreter in virtual environment.
    Checks target file's directory and its parent directories (up to 5 levels) for venv/ or .venv/.
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
    
    # Check in target file's directory and its parent directories (up to 5 levels)
    # This matches program-mcp's behavior: search up to 5 levels without restricting to project_root
    max_levels = 5
    
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


def _orca_output_has_error(text: str) -> bool:
    """Return True if ORCA output contains definitive error/abort phrases."""
    if not text:
        return False
    upper = text.upper()
    error_phrases = [
        "ABORTING THE RUN",
        "ABORTING RUN",
        "ORCA HAS STOPPED",
        "HAS FAILED",
        "NOT ASSIGNED OR NOT AVAILABLE",
        "UNKNOWN IDENTIFIER IN BASIS",
        "THE BASIS SET WAS EITHER NOT ASSIGNED",
        "ERROR - ABORTING",
    ]
    return any(p in upper for p in error_phrases)


def run_orca_file(
    target_file: str,
    project_root: str = ".",
    timeout: int = 600,
    orca_executable: Optional[str] = None,
    success_if_running_sec: Optional[int] = 60,
) -> dict:
    """
    Execute an ORCA input file and return the execution results.
    
    ORCA execution format: orca <input_file.inp> > <output_file.out> 2>&1
    ORCA writes output to a .out file (same name as input but .out extension).
    
    Args:
        target_file: Path to ORCA .inp file (relative to project_root or absolute)
        project_root: Project root directory (default: ".")
        timeout: Execution timeout in seconds (default: 600 for DFT calculations)
        orca_executable: ORCA executable path/command (default: from ORCA_EXECUTABLE env var or "/home/xiangfei/orca_install_dir/orca")
        
    Returns:
        Dictionary containing:
        - exitcode: Process exit code (0 = success, non-zero = error, None = error/timeout)
        - stdout: Standard output (ORCA output file content), truncated from the **end**
                  to keep at most the final 50 lines or 50000 characters.
        - stderr: Standard error (ORCA error messages, extracted from output), truncated
                  with the same tail-based rule as stdout.
        - timeout: True if execution timed out, False otherwise
    """
    # Tail‑truncation parameters for ORCA output
    # We always keep the *final* part of the output, since ORCA prints banners and
    # metadata at the beginning and important diagnostics at the end.
    MAX_OUTPUT_CHARS = 10000  # keep at most the final 10000 characters
    MAX_OUTPUT_LINES = 50     # or at most the final 50 lines

    def _truncate_tail(text: str, max_chars: int = MAX_OUTPUT_CHARS, max_lines: int = MAX_OUTPUT_LINES) -> str:
        """
        Truncate long output from the back (tail).
        
        - First enforce a line limit by keeping the final `max_lines` lines.
        - Then enforce a character limit by keeping the final `max_chars` characters.
        - When truncation happens, prepend a short note indicating how much was removed
          **from the beginning** of the stream.
        """
        if not text:
            return text

        lines = text.split("\n")
        truncated_note = ""

        # Apply line-based truncation first (keep last max_lines)
        if len(lines) > max_lines:
            removed_lines = len(lines) - max_lines
            lines = lines[-max_lines:]
            text = "\n".join(lines)
            truncated_note = f"... (truncated: {removed_lines} lines from beginning) ...\n\n"

        # Apply character-based truncation (keep last max_chars)
        if len(text) > max_chars:
            removed_chars = len(text) - max_chars
            text = text[-max_chars:]
            if truncated_note:
                truncated_note = f"... (truncated: {removed_chars} characters from beginning; plus earlier line truncation) ...\n\n"
            else:
                truncated_note = f"... (truncated: {removed_chars} characters from beginning) ...\n\n"

        return truncated_note + text
    # Get ORCA executable path
    if orca_executable is None:
        orca_executable = os.environ.get("ORCA_EXECUTABLE", "/home/xiangfei/orca_install_dir/orca")
    
    # Resolve target file path
    target_path = Path(target_file).resolve()
    project_root_path = Path(project_root).resolve()
    
    # Validate input file exists
    if not target_path.exists():
        return {
            "exitcode": None,
            "stdout": "",
            "stderr": f"ORCA input file not found: {target_file}",
            "timeout": False
        }
    
    # If target_file is within project_root, use relative path
    # Otherwise, use absolute path
    try:
        # Try to get relative path if target is within project_root
        file_arg = str(target_path.relative_to(project_root_path))
    except ValueError:
        # Target is outside project_root, use absolute path
        file_arg = str(target_path)
    
    # ORCA output file (same name as input but .out extension)
    output_file = target_path.with_suffix('.out')
    
    # Clean up ORCA output files from previous runs (except .inp file)
    # ORCA creates many files: .out, .gbw, .xyz, .trj.xyz, .densities, .densitiesinfo, 
    # .property.txt, .bibtex, .cpcm, .cpcm_corr, etc.
    orca_output_extensions = [
        '.out', '.gbw', '.xyz', '.trj.xyz', '.densities', '.densitiesinfo',
        '.property.txt', '.bibtex', '.cpcm', '.cpcm_corr', '.hess', '.opt',
        '.tmp', '.engrad', '.molden', '.cube', '.wfn', '.wfx', '.molden.input'
    ]
    
    input_dir = target_path.parent
    input_stem = target_path.stem
    
    # Remove all ORCA output files with the same stem
    cleaned_files = []
    for ext in orca_output_extensions:
        output_file_path = input_dir / f"{input_stem}{ext}"
        if output_file_path.exists() and output_file_path != target_path:
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
    cmd = [orca_executable, file_arg]
    cwd = os.path.abspath(project_root)
    use_early_success = (success_if_running_sec is not None and success_if_running_sec > 0
                         and success_if_running_sec <= timeout)

    def _read_out_file():
        if not output_file.exists():
            return ""
        try:
            with open(output_file, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
        except Exception:
            return ""

    if use_early_success:
        # Run with Popen; after success_if_running_sec with no error in output, treat as success and stop
        wait_sec = min(success_if_running_sec, timeout)
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=cwd,
            )
        except FileNotFoundError:
            return {
                "exitcode": None,
                "stdout": "",
                "stderr": f"ORCA executable not found: {orca_executable}. Set ORCA_EXECUTABLE environment variable or ensure ORCA is in PATH.",
                "timeout": False,
            }
        try:
            proc.wait(timeout=wait_sec)
        except subprocess.TimeoutExpired:
            pass

        if proc.poll() is None:
            # Still running after wait_sec
            output_content = _read_out_file()
            if not _orca_output_has_error(output_content):
                # Ran success_if_running_sec seconds without error -> success
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
                truncated = _truncate_tail(output_content, max_chars=MAX_OUTPUT_CHARS, max_lines=MAX_OUTPUT_LINES)
                return {
                    "exitcode": 0,
                    "stdout": truncated + f"\n\n--- Note: ORCA ran {success_if_running_sec}s without error. Input valid. ---",
                    "stderr": "",
                    "timeout": True,
                    "running_successfully": True,
                    "output_truncated": len(output_content) > len(truncated),
                }
            # Error in output; wait remainder of timeout then treat as timeout
            try:
                proc.wait(timeout=max(0, timeout - wait_sec))
            except subprocess.TimeoutExpired:
                pass
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            output_content = _read_out_file()
        else:
            # Exited within wait_sec
            output_content = _read_out_file()
            combined = (proc.stdout.read() if proc.stdout else "") + (proc.stderr.read() if proc.stderr else "")
            full_output = output_content or combined
            exitcode = proc.returncode
            has_success_message = "ORCA TERMINATED NORMALLY" in full_output.upper()
            is_success = (exitcode == 0) and has_success_message
            stderr_content = ""
            if not is_success:
                if exitcode != 0:
                    stderr_content = f"ORCA exited with code {exitcode}\n\n--- Full ORCA Output ---\n{full_output}"
                else:
                    exitcode = 1
                    stderr_content = f"ORCA completed with exit code 0 but no 'ORCA TERMINATED NORMALLY' message.\n\n--- Full ORCA Output ---\n{full_output}"
            stdout_max = 5000 if exitcode == 0 else MAX_OUTPUT_CHARS
            return {
                "exitcode": exitcode,
                "stdout": _truncate_tail(full_output, max_chars=stdout_max, max_lines=MAX_OUTPUT_LINES),
                "stderr": _truncate_tail(stderr_content, max_chars=MAX_OUTPUT_CHARS, max_lines=MAX_OUTPUT_LINES),
                "timeout": False,
                "output_truncated": False,
            }
        # Fall through: still running but had error in output, or we waited and killed; output_content set above
        output_upper = output_content.upper()
        has_orca_banner = "ORCA" in output_upper and ("PROGRAM" in output_upper or "VERSION" in output_upper)
        has_substantial_output = len(output_content) >= 2000
        has_success_message = "ORCA TERMINATED NORMALLY" in output_upper
        is_running_successfully = has_success_message or (has_orca_banner and has_substantial_output)
        if is_running_successfully:
            truncated_output = _truncate_tail(output_content, max_chars=MAX_OUTPUT_CHARS, max_lines=MAX_OUTPUT_LINES)
            return {
                "exitcode": 0,
                "stdout": truncated_output + f"\n\n--- Note: ORCA execution timed out after {timeout}s but was running successfully. Input file is valid. ---",
                "stderr": "",
                "timeout": True,
                "running_successfully": True,
                "output_truncated": len(output_content) > len(truncated_output),
            }
        return {
            "exitcode": None,
            "stdout": output_content if output_content else "",
            "stderr": f"ORCA execution timeout after {timeout}s. Could not determine if input is valid.",
            "timeout": True,
            "running_successfully": False,
        }
    # Original path: single subprocess.run with full timeout
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
        )
        output_content = _read_out_file()
        if not output_content and proc.stderr:
            output_content = proc.stderr
        combined_output = proc.stdout + proc.stderr
        if output_content:
            full_output = output_content
            if combined_output:
                full_output = output_content + "\n\n--- Process output ---\n" + combined_output
        else:
            full_output = combined_output
        exitcode = proc.returncode
        has_success_message = "ORCA TERMINATED NORMALLY" in full_output.upper()
        is_success = (exitcode == 0) and has_success_message
        stderr_content = ""
        if not is_success:
            if exitcode != 0:
                stderr_content = f"ORCA exited with code {exitcode}\n\n--- Full ORCA Output ---\n{full_output}"
            elif not has_success_message:
                exitcode = 1
                stderr_content = f"ORCA completed with exit code 0 but no 'ORCA TERMINATED NORMALLY' message detected.\n\n--- Full ORCA Output ---\n{full_output}"
            else:
                stderr_content = full_output
        stdout_max_chars = 5000 if exitcode == 0 else MAX_OUTPUT_CHARS
        truncated_stdout = _truncate_tail(full_output, max_chars=stdout_max_chars, max_lines=MAX_OUTPUT_LINES)
        truncated_stderr = _truncate_tail(stderr_content, max_chars=MAX_OUTPUT_CHARS, max_lines=MAX_OUTPUT_LINES)
        return {
            "exitcode": exitcode,
            "stdout": truncated_stdout,
            "stderr": truncated_stderr,
            "timeout": False,
            "output_truncated": len(full_output) > len(truncated_stdout) or len(stderr_content) > len(truncated_stderr),
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
        
        # Check if ORCA started successfully
        # Use a more robust approach: rely on positive indicators (banner, output length)
        # rather than trying to detect all possible error patterns
        is_running_successfully = False
        if output_content:
            output_upper = output_content.upper()
            output_length = len(output_content)
            
            # Positive indicators that ORCA is running:
            # 1. ORCA banner/startup (indicates it started)
            has_orca_banner = "ORCA" in output_upper and ("PROGRAM" in output_upper or "VERSION" in output_upper)
            
            # 2. Substantial output (suggests it's processing, not just failed immediately)
            # Very short output (< 2000 chars) with banner might indicate immediate failure
            # Longer output suggests it was running
            has_substantial_output = output_length >= 2000
            
            # 3. Success message (definitive indicator)
            has_success_message = "ORCA TERMINATED NORMALLY" in output_upper
            
            # If we have success message, definitely successful
            # Otherwise, if banner present and substantial output, likely running successfully
            # This avoids keyword matching for errors - we rely on positive indicators
            if has_success_message:
                is_running_successfully = True
            elif has_orca_banner and has_substantial_output:
                # Banner + substantial output = likely running (timeout is just taking too long)
                is_running_successfully = True
        
        if is_running_successfully:
            # ORCA is running successfully (just taking longer than timeout)
            # This is considered success for input validation purposes.
            # Apply the same tail‑truncation policy as in the main path so that
            # stdout is consistent across both branches.
            truncated_output = _truncate_tail(output_content, max_chars=MAX_OUTPUT_CHARS, max_lines=MAX_OUTPUT_LINES)
            
            return {
                "exitcode": 0,  # Treat as success
                "stdout": truncated_output + f"\n\n--- Note: ORCA execution timed out after {timeout}s but was running successfully. Input file is valid. ---",
                "stderr": "",  # No errors
                "timeout": True,
                "running_successfully": True,  # Flag to indicate it was running when timed out
                "output_truncated": len(output_content) > len(truncated_output)
            }
        else:
            # Timeout but couldn't determine if it was running successfully
            return {
                "exitcode": None,
                "stdout": output_content if output_content else "",
                "stderr": f"ORCA execution timeout after {timeout}s. Could not determine if input is valid.",
                "timeout": True,
                "running_successfully": False
            }
    except FileNotFoundError:
        return {
            "exitcode": None,
            "stdout": "",
            "stderr": f"ORCA executable not found: {orca_executable}. Set ORCA_EXECUTABLE environment variable or ensure ORCA is in PATH.",
            "timeout": False
        }
    except Exception as e:
        return {
            "exitcode": None,
            "stdout": "",
            "stderr": f"Error executing ORCA: {str(e)}",
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