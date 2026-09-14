#!/usr/bin/env python3
"""
Extract XYZ structures from LaTeX files in the quantum_chemistry directory.
Output files are named: {paper_name}_{structure_name}.xyz
"""

import re
import os
from pathlib import Path


# Valid chemical element symbols (1–2 letters) for XYZ
ELEMENT_PATTERN = r'([A-Z][a-z]?)'
# One line: optional spaces, element, spaces, three numbers (with optional minus/decimals)
COORD_LINE_INLINE = re.compile(
    r'^\s*' + ELEMENT_PATTERN + r'\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*$'
)
# Table format: "N Element x y z" on one line
COORD_LINE_TABLE = re.compile(
    r'^\s*(\d+)\s+' + ELEMENT_PATTERN + r'\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*$'
)
# Structure name in brackets e.g. [CuI(NHC2)]+
STRUCTURE_NAME_BRACKETS = re.compile(r'^\[([^\]]+)\]\s*[\d+]*\s*$')
# Supplementary table header: "Nuclear coordinates of S1 geometry of BNOO in gas phase"
# Capture molecule name (e.g. BNOO, BNSS) after "geometry of"
TABLE_HEADER = re.compile(
    r'Supplementary Table \d+\s*\|\s*Nuclear coordinates of (?:S1 geometry of )?([A-Za-z0-9]+)',
    re.IGNORECASE
)
TABLE_HEADER_ALT = re.compile(
    r'Nuclear coordinates of (?:S1 geometry of )?([A-Za-z0-9]+)',
    re.IGNORECASE
)


def sanitize_filename(s: str) -> str:
    """Make a string safe for use as filename (no path chars, no brackets)."""
    s = s.strip()
    for c in r'[]()+,\\/:*?"<>|':
        s = s.replace(c, '_')
    s = re.sub(r'_+', '_', s)
    return s[:80].strip('_') or 'structure'


def extract_inline_xyz_blocks(content: str):
    """
    Find blocks of inline XYZ coordinates (element x y z per line).
    Preceding line may be structure name like [CuI(NHC2)]+.
    """
    lines = content.splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Check for structure name line (bracket form)
        name_match = STRUCTURE_NAME_BRACKETS.match(line.strip())
        if name_match:
            struct_name = sanitize_filename(name_match.group(1))
            coords = []
            j = i + 1
            # Skip "Charge = ..." etc., but only within a short window (real coords follow soon)
            lookahead = 15
            while j < len(lines) and j - i <= lookahead and not COORD_LINE_INLINE.match(lines[j].strip()):
                j += 1
            if j - i > lookahead:
                # No coords found nearby; this was a mention, not a coord block header
                i += 1
                continue
            while j < len(lines):
                m = COORD_LINE_INLINE.match(lines[j].strip())
                if m:
                    elem, x, y, z = m.group(1), m.group(2), m.group(3), m.group(4)
                    coords.append((elem, x, y, z))
                    j += 1
                else:
                    stripped = lines[j].strip()
                    # Next structure name: stop and keep this block
                    if STRUCTURE_NAME_BRACKETS.match(stripped):
                        break
                    # Non-coord line while we already have coords: stop
                    if coords and stripped and not re.match(r'^[A-Z][a-z]?\s+[-\d.]', stripped):
                        break
                    # No coords yet: skip this line (e.g. "Charge = ...") and keep looking
                    j += 1
            if coords:
                blocks.append((struct_name, coords))
            i = j
            continue
        i += 1
    return blocks


def extract_table_xyz_blocks(content: str):
    """
    Find tables of nuclear coordinates (Supplementary Table N | Nuclear coordinates of ...).
    Handles both compact (N Element x y z on one line) and split (number, element, x, y, z on separate lines).
    """
    blocks = []
    # Find all table headers and extract structure name
    for header_match in TABLE_HEADER.finditer(content):
        struct_name = sanitize_filename(header_match.group(1))
        start = header_match.end()
        # Find the next "Supplementary Table" or end of reasonable block (e.g. 2000 chars)
        next_table = TABLE_HEADER.search(content[start : start + 50000])
        if next_table:
            block_end = start + next_table.start()
        else:
            block_end = min(start + 50000, len(content))
        block = content[start:block_end]
        coords = parse_table_coords(block)
        if coords:
            blocks.append((struct_name, coords))
    # Also try alternate header (without "Supplementary Table N |")
    for header_match in TABLE_HEADER_ALT.finditer(content):
        struct_name = sanitize_filename(header_match.group(1))
        start = header_match.end()
        next_table = TABLE_HEADER_ALT.search(content[start : start + 50000])
        if next_table:
            block_end = start + next_table.start()
        else:
            block_end = min(start + 50000, len(content))
        block = content[start:block_end]
        coords = parse_table_coords(block)
        if coords and (struct_name, coords) not in [(b[0], b[1]) for b in blocks]:
            blocks.append((struct_name, coords))
    return blocks


def parse_table_coords(block: str):
    """Parse coordinate lines from a table block. Handles compact and split formats."""
    lines = block.splitlines()
    coords = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Compact: "1 C -0.03 4.83 0.42"
        m = COORD_LINE_TABLE.match(line.strip())
        if m:
            elem, x, y, z = m.group(2), m.group(3), m.group(4), m.group(5)
            coords.append((elem, x, y, z))
            i += 1
            continue
        # Split: number on one line, element on next, x, y, z on following lines
        if line.strip().isdigit() and i + 4 <= len(lines):
            try:
                elem_line = lines[i + 1].strip()
                x_line = lines[i + 2].strip()
                y_line = lines[i + 3].strip()
                z_line = lines[i + 4].strip()
                elem_match = re.match(r'^([A-Z][a-z]?)\s*$', elem_line)
                if elem_match and re.match(r'^[-\d.]+$', x_line) and re.match(r'^[-\d.]+$', y_line) and re.match(r'^[-\d.]+$', z_line):
                    coords.append((elem_match.group(1), x_line, y_line, z_line))
                    i += 5
                    continue
            except (IndexError, AttributeError):
                pass
        i += 1
    return coords


def write_xyz_file(path: Path, structure_name: str, coords: list):
    """Write standard XYZ file: line1=count, line2=comment, then Element x y z."""
    with open(path, 'w') as f:
        f.write(str(len(coords)) + '\n')
        f.write(structure_name + '\n')
        for elem, x, y, z in coords:
            f.write(f'{elem} {x} {y} {z}\n')


def main():
    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / 'extracted_xyz'
    out_dir.mkdir(exist_ok=True)

    tex_files = sorted(base_dir.glob('*.tex'))
    total_written = 0
    seen = set()  # (paper, struct_name) to avoid duplicates

    for tex_path in tex_files:
        paper_name = tex_path.stem
        try:
            content = tex_path.read_text(encoding='utf-8', errors='replace')
        except Exception as e:
            print(f'Skip {tex_path.name}: {e}')
            continue

        # Inline blocks (e.g. ja1c09505_s1)
        for struct_name, coords in extract_inline_xyz_blocks(content):
            key = (paper_name, struct_name)
            if key in seen or len(coords) < 3:
                continue
            seen.add(key)
            fname = f'{paper_name}_{struct_name}.xyz'
            out_path = out_dir / fname
            write_xyz_file(out_path, struct_name, coords)
            print(f'  {fname} ({len(coords)} atoms)')
            total_written += 1

        # Table blocks (e.g. s41467)
        for struct_name, coords in extract_table_xyz_blocks(content):
            key = (paper_name, struct_name)
            if key in seen or len(coords) < 3:
                continue
            seen.add(key)
            fname = f'{paper_name}_{struct_name}.xyz'
            out_path = out_dir / fname
            write_xyz_file(out_path, struct_name, coords)
            print(f'  {fname} ({len(coords)} atoms)')
            total_written += 1

    print(f'\nTotal: {total_written} XYZ files written to {out_dir}')


if __name__ == '__main__':
    main()
