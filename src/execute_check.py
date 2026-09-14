# execute_check: run all code_LLM_loopx programs in check_dir concurrently, write report_execute_check.jsonl
import asyncio
import json
import re
from pathlib import Path

from utils import run_program

_CODE_LOOP_PATTERN = re.compile(r"^code_LLM_loop\d+\.(py|jl)$")


def _collect_loop_files(check_dir: Path) -> list[Path]:
    """Collect code_LLM_loop{N}.py or .jl files, sorted by loop number."""
    files = []
    for f in check_dir.iterdir():
        if f.is_file() and _CODE_LOOP_PATTERN.match(f.name):
            files.append(f)
    files.sort(key=lambda p: (int(re.search(r"\d+", p.stem).group()), p.name))
    return files


async def run_execute_check_async(
    check_dir: str | Path,
    timeout: int = 300,
    concurr_num: int = 8,
) -> str:
    """
    Execute all code_LLM_loopx programs in check_dir concurrently.
    Write results to report_execute_check.jsonl.

    Returns:
        Path to report_execute_check.jsonl
    """
    check_path = Path(check_dir)
    if not check_path.exists() or not check_path.is_dir():
        raise FileNotFoundError(f"Check directory not found: {check_path}")

    loop_files = _collect_loop_files(check_path)
    if not loop_files:
        report_path = check_path / "report_execute_check.jsonl"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("")
        return str(report_path)

    sem = asyncio.Semaphore(concurr_num)

    async def _run_one(target_path: Path) -> dict:
        async with sem:
            ext = target_path.suffix.lower()
            if ext == ".py":
                run_func = run_program.run_python_file
            elif ext == ".jl":
                run_func = run_program.run_julia_file
            else:
                raise ValueError(f"Unsupported extension: {ext}")
            result = await asyncio.to_thread(
                run_func, str(target_path), str(check_path), timeout
            )
            return {
                "name": target_path.name,
                "exitcode": result.get("exitcode"),
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
            }

    tasks = [asyncio.create_task(_run_one(p)) for p in loop_files]
    results = await asyncio.gather(*tasks)

    report_path = check_path / "report_execute_check.jsonl"
    with open(report_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return str(report_path)
