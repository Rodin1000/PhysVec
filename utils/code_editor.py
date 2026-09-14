import os, re, ast
import libcst as cst
from pathlib import Path
from typing import List, Optional

def _fix_julia_type_annotations(src: str) -> str:
    """
    Fix Julia-style type annotations (::) to Python style (:) in function parameters.
    Example: def func(x::int) -> def func(x: int)
    """
    # Match pattern: parameter_name::type in function parameter lists
    # This regex matches ::type in function definitions
    return re.sub(r'(\w+)::(\w+)', r'\1: \2', src)


def _remove_julia_bracket_contents(s: str) -> str:
    """
    Replace content inside balanced square brackets with nothing (leave '[]').
    So 'end' inside indexing (e.g. F[1:2:end]) is not counted as block-closing 'end'.
    """
    out = []
    i = 0
    while i < len(s):
        if s[i] == '[':
            out.append('[')
            i += 1
            depth = 1
            while i < len(s) and depth > 0:
                if s[i] == '[':
                    depth += 1
                elif s[i] == ']':
                    depth -= 1
                i += 1
            out.append(']')
        else:
            out.append(s[i])
            i += 1
    return ''.join(out)


def split_functions(src_path: str, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    ext = os.path.splitext(src_path)[1].lower()
    with open(src_path, "r", encoding="utf-8") as f:
        src = f.read()
    lines = src.splitlines()

    if ext == ".py":
        print("repository generating...")
        
        # Parse the source code first
        try:
            tree = ast.parse(src)
        except SyntaxError:
            # Attempt to fix common syntax errors (e.g., Julia-style :: annotations)
            src = _fix_julia_type_annotations(src)
            lines = src.splitlines()  # Update lines after fixing
            tree = ast.parse(src)
        
        # Extract all top-level import statements using AST
        import_nodes = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_nodes.append(node)
        
        # Sort by line number to maintain original order
        import_nodes.sort(key=lambda n: n.lineno)
        
        # Extract import statements from original source code
        import_statements = []
        for node in import_nodes:
            # Get the line number (1-based)
            start_line = node.lineno - 1  # Convert to 0-based index
            # Get end line number if available, otherwise use start_line
            end_line = getattr(node, "end_lineno", node.lineno) - 1
            # Extract the lines from original source
            for i in range(start_line, end_line + 1):
                if i < len(lines):
                    import_statements.append(lines[i])
        
        import_block = "\n".join(import_statements) + "\n\n" if import_statements else ""
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Find the start line (considering decorators)
                decorator_lines = []
                if node.decorator_list:
                    for decorator in node.decorator_list:
                        decorator_lineno = getattr(decorator, "lineno", node.lineno)
                        decorator_lines.append(decorator_lineno)
                start_line = min(decorator_lines + [node.lineno]) if decorator_lines else node.lineno
                start = start_line - 1  # Convert to 0-based index

                # Find the end line
                end_line = getattr(node, "end_lineno", None)
                if end_line is None:
                    # Fallback: find the last line of the function body
                    if node.body:
                        last_stmt = node.body[-1]
                        end_line = getattr(last_stmt, "end_lineno", getattr(last_stmt, "lineno", node.lineno))
                    else:
                        end_line = node.lineno
                end = end_line - 1  # Convert to 0-based index

                # Extract metadata comments (Input/Output) immediately preceding the function
                metas = []
                meta_idx = start - 1
                while meta_idx >= 0 and lines[meta_idx].strip().startswith("#"):
                    meta_idx -= 1
                meta_idx += 1

                if meta_idx < start:
                    # Found some comment lines
                    for idx in range(meta_idx, start):
                        stripped = lines[idx].strip()
                        if re.match(r"#\s*Input", stripped) or re.match(r"#\s*Output", stripped):
                            metas.append(lines[idx].rstrip())

                # Extract function code (including decorators, signature, and body)
                function_body = "\n".join(metas + [l.rstrip() for l in lines[start:end+1]]) + "\n"
                out = import_block + function_body
                
                name = node.name
                with open(os.path.join(dst_dir, f"{name}.py"), "w", encoding="utf-8") as g:
                    g.write(out)
        return

    if ext == ".jl":
        print("repository generating...")
        # Extract using statements and top-level struct blocks (index-based)
        i, n = 0, len(lines)
        using_lines = []
        while i < n:
            line = lines[i]
            s = line.strip()
            if s.startswith("using ") or s.startswith("import "):
                using_lines.append(line)
                i += 1
            elif s == "" or s.startswith("#"):
                i += 1
            else:
                break
        using_block = "\n".join(using_lines) + "\n\n" if using_lines else ""

        # Extract top-level struct/mutable struct blocks before first function
        struct_blocks = []
        while i < n:
            line = lines[i]
            s = line.strip()
            if s == "" or s.startswith("#"):
                i += 1
                continue
            if re.match(r"^\s*function\s+", line):
                break
            if re.match(r"^\s*(mutable\s+)?struct\s+", line):
                start = i
                j = i
                nesting = 1
                j += 1
                open_kw = r"\b(function|begin|if|for|while|let|try|struct|quote|module|baremodule|do|mutable\s+struct)\b"
                while j < n:
                    cl = lines[j]
                    clean = re.sub(r'"[^"]*"', '', cl)
                    clean = re.sub(r"#.*", "", clean)
                    clean = re.sub(r'\[.*?\bfor\b.*?\]', '', clean)
                    clean = _remove_julia_bracket_contents(clean)
                    if j > start:
                        nesting += len(re.findall(open_kw, clean))
                    nesting -= len(re.findall(r"\bend\b", clean))
                    if nesting == 0:
                        struct_blocks.append("\n".join(l.rstrip() for l in lines[start:j + 1]))
                        i = j + 1
                        break
                    j += 1
                else:
                    i += 1
            else:
                i += 1
        preamble = using_block + ("\n\n".join(struct_blocks) + "\n\n" if struct_blocks else "")
        i = 0
        while i < n:
            line = lines[i]
            stripped = line.strip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith("#"):
                i += 1
                continue
            
            # Match 'function' at the start of a line, allowing for whitespace.
            # Captures function name.
            m = re.match(r"^\s*function\s+([A-Za-z_][\w\d_]*!?)\s*\(", line)
            if not m:
                i += 1
                continue
            
            name, start = m.group(1), i
            metas = []
            # Look for metadata comments immediately preceding the function
            meta_idx = start - 1
            while meta_idx >= 0 and lines[meta_idx].strip().startswith("#"):
                meta_idx -= 1
            meta_idx += 1
            
            if meta_idx < start:
                # Found some comment lines
                for idx in range(meta_idx, start):
                    if re.match(r"\s*#\s*Input", lines[idx]) or re.match(r"\s*#\s*Output", lines[idx]):
                        metas.append(lines[idx].rstrip())

            # Robustly find the matching 'end' for multi-line functions
            j = start
            nesting_level = 1  # Start with 1 because we've already seen 'function'
            
            while j < n:
                current_line = lines[j]
                # Remove string literals and comments to avoid false matches
                clean_line = re.sub(r'"[^"]*"', '', current_line)
                clean_line = re.sub(r"#.*", "", clean_line)
                
                # Remove list/array comprehensions [... for ... in ...] to avoid counting 'for' in them
                clean_line = re.sub(r'\[.*?\bfor\b.*?\]', '', clean_line)
                clean_line = _remove_julia_bracket_contents(clean_line)
                
                # Only count keywords after the first line (to avoid counting the initial 'function')
                if j > start:
                    # Keywords that open a block
                    open_keywords = r"\b(function|begin|if|for|while|let|try|struct|quote|module|baremodule|do|mutable\s+struct)\b"
                    
                    # Count opening keywords
                    opens = re.findall(open_keywords, clean_line)
                    nesting_level += len(opens)
                
                # Count closing 'end' keywords (on all lines including start)
                ends = re.findall(r"\bend\b", clean_line)
                nesting_level -= len(ends)
                
                # When nesting level returns to 0, we found the function's end
                if nesting_level == 0:
                    # Found the end of the function
                    function_body = "\n".join(metas + [l.rstrip() for l in lines[start:j+1]]) + "\n"
                    out = preamble + function_body
                    with open(os.path.join(dst_dir, f"{name}.jl"), "w", encoding="utf-8") as g:
                        g.write(out)
                    i = j + 1
                    break
                j += 1
            else:
                # Did not find a matching end, skip this line
                i += 1
        return

    raise ValueError("Unsupported file extension; expected .jl or .py")



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
        # Parse error (e.g. syntax error): skip this file, do not remove imports
        return
    
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
                # Recursively collect all names in the attribute chain
                # For "x.function", collect "x"
                # For "x.yy.function", collect "x" (from x.yy)
                current = node.value
                while isinstance(current, cst.Attribute):
                    # Collect the intermediate name (e.g., "x" in "x.yy")
                    if isinstance(current.value, cst.Name):
                        self.used_names.add(current.value.value)
                    current = current.value
                # Collect the base name (e.g., "x" in "x.function" or final name in chain)
                if isinstance(current, cst.Name):
                    self.used_names.add(current.value)


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