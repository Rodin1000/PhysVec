#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import os, re, ast
import libcst as cst
from pathlib import Path
from typing import List, Optional,Tuple


# Matches lines like:
#   #Tags_keywords
#   #Tags_inputblocks_basis
#   #Tags_geometryblocks
TAG_LINE_RE = re.compile(
    r"^\s*#Tags_(keywords|inputblocks|geometryblocks)(?:_([A-Za-z0-9]+))?\s*$"
)

SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe(s: str) -> str:
    s2 = SAFE_NAME_RE.sub("_", (s or "").strip())
    return s2.strip("_") or "none"


def _find_tag_starts(lines: List[str]) -> List[Tuple[int, str, str]]:
    starts: List[Tuple[int, str, str]] = []
    for i, ln in enumerate(lines):
        m = TAG_LINE_RE.match(ln)
        if not m:
            continue
        block_type = m.group(1)
        sub_type = m.group(2) or ""
        # Per your convention, inputblocks should have a sub-type; if missing, label it.
        if block_type == "inputblocks" and not sub_type:
            sub_type = "unknown"
        starts.append((i, block_type, sub_type))
    return starts


def _slice_chunks(lines: List[str], starts: List[Tuple[int, str, str]]) -> List[Tuple[int, int, str, str]]:
    if not starts:
        return []
    starts_sorted = sorted(starts, key=lambda x: x[0])
    chunks: List[Tuple[int, int, str, str]] = []
    for k, (s, bt, sub) in enumerate(starts_sorted):
        e = starts_sorted[k + 1][0] if k + 1 < len(starts_sorted) else len(lines)
        chunks.append((s, e, bt, sub))
    return chunks


def split_functions(
    src_path: str,
    dst_dir: str,
    prefix: str = "seg",
    keep_line_endings: bool = True,
) -> List[Path]:
    """
    Split an ORCA input text (e.g., code_LLM.inp) into smaller segments by '#Tags_...' markers.

    Each output file contains exactly one tag block:
      - the '#Tags_...' line
      - its following comment + ORCA lines
      - up to (but not including) the next '#Tags_...' line

    Output filenames:
      <prefix>_###__<block_type>__<sub_or_none>.inp
    """
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src_path}")

    os.makedirs(dst_dir, exist_ok=True)
    out_dir = Path(dst_dir)

    text = src.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=keep_line_endings)

    starts = _find_tag_starts(lines)
    chunks = _slice_chunks(lines, starts)

    if not chunks:
        raise ValueError("No '#Tags_...' lines found. Cannot split by tags.")

    written: List[Path] = []
    for idx, (s, e, bt, sub) in enumerate(chunks, start=1):
        seg_lines = lines[s:e]
        if not any(ln.strip() for ln in seg_lines):
            continue

        fname = f"{_safe(bt)}_{_safe(sub)}.inp"
        out_path = out_dir / fname

        content = "".join(seg_lines).rstrip() + ("\n" if keep_line_endings else "")
        out_path.write_text(content, encoding="utf-8")
        written.append(out_path)

    return written







def delete_unused_imports(
    dst_dir: str,
    count_type_annotations: bool = True,
    file_names: Optional[List[str]] = None
) -> None:
    """
    Remove unused import statements from all .py and .jl files in the target directory.
    Currently only Python files are processed, Julia files are skipped.
    
    Args:
        dst_dir: Target directory path
        count_type_annotations: If True (default), names in type annotations are considered as "used".
                               If False, names in type annotations are not considered as "used".
        file_names: If None (default), process all files in the directory.
                   If a list of file names is provided, only process files whose names (with extension) are in the list.
                   Example: ["file1.py", "file2.py"]
    """
    dst_path = Path(dst_dir)
    if not dst_path.exists():
        raise FileNotFoundError(f"Directory {dst_dir} does not exist")
    
    # Process all Python and Julia files in the directory
    for file_path in dst_path.rglob("*"):
        if not file_path.is_file():
            continue
        
        # Filter by file_names if specified
        if file_names is not None:
            if file_path.name not in file_names:
                continue
        
        ext = file_path.suffix.lower()
        if ext == ".py":
            _process_python_file(file_path, count_type_annotations)
        elif ext == ".jl":
            # Julia file processing - placeholder for future implementation
            pass


def _process_python_file(file_path: Path, count_type_annotations: bool = True) -> None:
    """Process a single Python file and remove unused imports."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source_code = f.read()
    except Exception:
        raise ValueError(f"Failed to read file {file_path}")
    
    try:
        module = cst.parse_module(source_code)
    except Exception:
        # Skip files that cannot be parsed
        raise ValueError(f"Failed to parse file {file_path}")
    
    # Collect all top-level imports (outside function definitions)
    import_collector = ImportCollector()
    module.visit(import_collector)
    
    # Collect all used names (optionally excluding type annotations)
    usage_collector = UsageCollector(count_type_annotations=count_type_annotations)
    module.visit(usage_collector)
    
    # Determine which imports are unused or partially used
    used_names = usage_collector.used_names
    imports_to_remove_completely = []
    imports_to_modify = []  # List of (import_node, names_to_keep)
    
    # Debug: collect all imported names and used imported names
    all_imported_names = []
    used_imported_names = set()
    imports_to_remove_info = []
    imports_to_modify_info = []
    
    for import_node, imported_names in import_collector.imports:
        # Debug: add to all imported names
        all_imported_names.extend(imported_names)
        
        # Check if this is a "from ... import *" statement
        if "*" in imported_names:
            # Keep import * statements as we cannot determine what they import
            continue
        
        # Check which imported names are used
        used_in_this_import = [name for name in imported_names if name in used_names]
        used_imported_names.update(used_in_this_import)
        
        if not used_in_this_import:
            # Completely unused, remove the entire statement
            imports_to_remove_completely.append(import_node)
            # Debug: get import statement code for display
            try:
                import_code = cst.Module([import_node]).code.strip()
                imports_to_remove_info.append(import_code)
            except Exception:
                imports_to_remove_info.append(str(import_node))
        elif len(used_in_this_import) < len(imported_names):
            # Partially used, keep only the used names
            imports_to_modify.append((import_node, set(used_in_this_import)))
            # Debug: get import statement code for display
            try:
                import_code = cst.Module([import_node]).code.strip()
                imports_to_modify_info.append((import_code, used_in_this_import))
            except Exception:
                imports_to_modify_info.append((str(import_node), used_in_this_import))
    
    # Debug: output information for this file
    print(f"\n=== Processing file: {file_path.name} ===")
    print(f"All imported names: {sorted(set(all_imported_names))}")
    print(f"Used imported names: {sorted(used_imported_names)}")
    print(f"Imports to remove completely ({len(imports_to_remove_completely)}):")
    for imp_info in imports_to_remove_info:
        print(f"  - {imp_info}")
    print(f"Imports to modify (keep only used names) ({len(imports_to_modify)}):")
    for imp_info, names in imports_to_modify_info:
        print(f"  - {imp_info} -> keep: {sorted(names)}")
    
    # Remove or modify unused imports
    if imports_to_remove_completely or imports_to_modify:
        transformer = ImportRemover(imports_to_remove_completely, imports_to_modify)
        modified_module = module.visit(transformer)
        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(modified_module.code)
            total_removed = len(imports_to_remove_completely) + len(imports_to_modify)
            print(f"Successfully processed {total_removed} import statement(s)")
        except Exception:
            raise ValueError(f"Failed to write file {file_path}")
    else:
        print("No unused imports to remove")
    

class ImportCollector(cst.CSTVisitor):
    """Collect all top-level import statements (outside function definitions)."""
    
    def __init__(self):
        self.imports = []  # List of (import_node, imported_names)
        self.in_function = False
    
    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        self.in_function = True
    
    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        self.in_function = False
    
    def visit_Import(self, node: cst.Import) -> None:
        if not self.in_function:
            imported_names = []
            for alias in node.names:
                if alias.asname:
                    imported_names.append(alias.asname.name.value)
                else:
                    imported_names.append(alias.name.value.split(".")[0])
            self.imports.append((node, imported_names))
    
    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if not self.in_function:
            imported_names = []
            for alias in node.names:
                if isinstance(alias, cst.ImportStar):
                    imported_names.append("*")
                elif alias.asname:
                    imported_names.append(alias.asname.name.value)
                else:
                    imported_names.append(alias.name.value)
            self.imports.append((node, imported_names))


class UsageCollector(cst.CSTVisitor):
    """Collect all names used in code (excluding names in import statements, optionally excluding type annotations)."""
    
    def __init__(self, count_type_annotations: bool = True):
        self.used_names = set()
        self.in_import = False
        self.in_annotation = False
        self.count_type_annotations = count_type_annotations
    
    def visit_Import(self, node: cst.Import) -> None:
        self.in_import = True
    
    def leave_Import(self, node: cst.Import) -> None:
        self.in_import = False
    
    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        self.in_import = True
    
    def leave_ImportFrom(self, node: cst.ImportFrom) -> None:
        self.in_import = False
    
    def visit_Annotation(self, node: cst.Annotation) -> None:
        self.in_annotation = True
    
    def leave_Annotation(self, node: cst.Annotation) -> None:
        self.in_annotation = False
    
    def visit_Name(self, node: cst.Name) -> None:
        # Exclude names in import statements
        if not self.in_import:
            # If count_type_annotations is False, exclude names in type annotations
            if self.count_type_annotations or not self.in_annotation:
                self.used_names.add(node.value)
    
    def visit_Attribute(self, node: cst.Attribute) -> None:
        # Exclude attributes in import statements
        if not self.in_import:
            # If count_type_annotations is False, exclude attributes in type annotations
            if self.count_type_annotations or not self.in_annotation:
                # For attributes like "np.array", collect "np"
                if isinstance(node.value, cst.Name):
                    self.used_names.add(node.value.value)


class ImportRemover(cst.CSTTransformer):
    """Remove or modify import nodes (complete removal or partial removal of unused names)."""
    
    def __init__(self, imports_to_remove_completely, imports_to_modify):
        # Use code string representation as keys for reliable matching
        # Completely remove these imports
        self.imports_to_remove_by_code = {}
        for node in imports_to_remove_completely:
            try:
                code_str = cst.Module([node]).code.strip()
                self.imports_to_remove_by_code[code_str] = node
            except Exception:
                pass
        
        # Partially modify these imports: keep only specified names
        self.imports_to_modify_by_code = {}
        for node, names_to_keep in imports_to_modify:
            try:
                code_str = cst.Module([node]).code.strip()
                self.imports_to_modify_by_code[code_str] = names_to_keep
            except Exception:
                pass
    
    def leave_Import(self, original_node: cst.Import, updated_node: cst.Import) -> cst.RemovalSentinel:
        # Match by code string representation
        try:
            code_str = cst.Module([original_node]).code.strip()
            if code_str in self.imports_to_remove_by_code:
                return cst.RemoveFromParent()
        except Exception:
            pass
        return updated_node
    
    def leave_ImportFrom(self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom) -> cst.RemovalSentinel:
        # Match by code string representation
        try:
            code_str = cst.Module([original_node]).code.strip()
            
            # Check if this import should be completely removed
            if code_str in self.imports_to_remove_by_code:
                return cst.RemoveFromParent()
            
            # Check if this import should be partially modified
            if code_str in self.imports_to_modify_by_code:
                names_to_keep = self.imports_to_modify_by_code[code_str]
                # Build new names list, keeping only the used names
                new_names = []
                for alias in original_node.names:
                    if isinstance(alias, cst.ImportStar):
                        # Keep import * as is
                        new_names.append(alias)
                    else:
                        # Get the name (either alias name or original name)
                        name = alias.asname.name.value if alias.asname else alias.name.value
                        if name in names_to_keep:
                            new_names.append(alias)
                
                if new_names:
                    # Ensure the last alias has no comma to avoid trailing comma
                    last_alias = new_names[-1]
                    if not isinstance(last_alias, cst.ImportStar) and last_alias.comma is not None:
                        new_names[-1] = last_alias.with_changes(comma=None)
                    return updated_node.with_changes(names=new_names)
                else:
                    # If no names to keep, remove the entire statement
                    return cst.RemoveFromParent()
        except Exception:
            pass
        return updated_node



# for debug or test
# def main():
#     split_functions("../logs/_PhysRevB.58.R14741_Fig._1_form1/code_LLM.jl", "../Output_repo/test")


# if __name__ == "__main__":
#     main()












# def main(argv: Optional[List[str]] = None) -> None:
#     import argparse

#     p = argparse.ArgumentParser(
#         description="Split a concatenated ORCA code_LLM.inp into multiple single-job .inp files."
#     )
#     p.add_argument("src", help="Path to code_LLM.inp (or any concatenated ORCA input text).")
#     p.add_argument("dst_dir", help="Output directory to write split .inp files.")
#     p.add_argument("--prefix", default="job", help="Output filename prefix (default: job).")

#     args = p.parse_args(argv)

#     written = split_orca_code_llm_inp(args.src, args.dst_dir, prefix=args.prefix)
#     print(f"Wrote {len(written)} file(s) to: {Path(args.dst_dir).resolve()}")
#     for pth in written:
#         print(f"  - {pth.name}")


# if __name__ == "__main__":
#     main()
