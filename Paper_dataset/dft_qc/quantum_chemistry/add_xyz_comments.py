#!/usr/bin/env python3
"""
Add comment line (line 2) to each XYZ in extracted_xyz/ with:
  paper title (from .tex), tex filename, and brief structure description.
Standard XYZ: line1=atom count, line2=comment, line3+= coordinates.
"""

import re
from pathlib import Path


def get_tex_stems(tex_dir: Path) -> set:
    """Return set of .tex file stems for matching."""
    return {p.stem for p in tex_dir.glob("*.tex")}


def parse_xyz_filename(stem: str, tex_stems: set) -> tuple:
    """
    From xyz stem like 'ja1c09505_s1_CuI_NHC2', return (paper_id, structure_name)
    by longest matching tex stem.
    """
    parts = stem.split("_")
    for i in range(len(parts), 0, -1):
        candidate = "_".join(parts[:i])
        if candidate in tex_stems:
            structure = "_".join(parts[i:]) if i < len(parts) else "structure"
            return candidate, structure
    return stem, "structure"


def extract_title_from_tex(tex_path: Path, max_title_chars: int = 200) -> str:
    """Extract paper/supplementary title from first pages of .tex (PDF-converted)."""
    if not tex_path.exists():
        return "N/A"
    try:
        text = tex_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return "N/A"
    lines = [ln.strip() for ln in text.splitlines()[:100]]
    skip_patterns = (
        r"^---\s*Page", r"^$", r"^S\d+$", r"^\d+$",
        r"^Supplementary Information", r"^Supporting Information",
        r"^SUPPORTING INFORMATION", r"^Electronic Supplementary",
        r"^Contents$", r"^email:", r"^Article$",
        r"^\\", r"^%", r"^Cite This:", r"^Read Online", r"^ACCESS", r"^Metrics",
    )
    title_lines = []
    for line in lines:
        if not line:
            if title_lines:
                break
            continue
        if any(re.match(p, line, re.IGNORECASE) for p in skip_patterns):
            continue
        if re.search(r"@|University|Institute|Laboratory|\.(edu|ac\.|org)|https://", line, re.IGNORECASE):
            if title_lines:
                break
            continue
        if len(line) > 120:
            if title_lines:
                break
            continue
        title_lines.append(line)
        if len(" ".join(title_lines)) >= max_title_chars or len(title_lines) >= 5:
            break
    title = " ".join(title_lines).strip() if title_lines else "N/A"
    if len(title) > max_title_chars:
        title = title[: max_title_chars - 3] + "..."
    return title or "N/A"


def structure_description(structure_name: str) -> str:
    """Brief description from structure name."""
    s = structure_name.replace("_", " ")
    if re.match(r"^S[0-3][AB]?$", structure_name, re.IGNORECASE):
        s = s + " (OEC QM model, oxygen-evolving complex)"
    elif "NHC2" in structure_name or "NHC4" in structure_name:
        s = s + " (N-heterocyclic carbene complex)"
    elif "CuI" in structure_name or "CuII" in structure_name or "CuIII" in structure_name:
        s = s + " (copper complex)"
    elif "BN" in structure_name and ("OO" in structure_name or "SS" in structure_name or "SeSe" in structure_name or "TeTe" in structure_name or "PoPo" in structure_name or "COCO" in structure_name):
        s = s + " (chalcogen-doped / MR-TADF molecule)"
    elif "CF3" in structure_name:
        s = s + " (Cu(CF3)4 anion)"
    return s.strip()


def add_comment_to_xyz(xyz_path: Path, comment: str) -> None:
    """Replace line 2 of XYZ with comment."""
    lines = xyz_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 2:
        return
    new_lines = [lines[0], comment] + lines[2:]
    xyz_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def main():
    base = Path(__file__).resolve().parent
    extracted_dir = base / "extracted_xyz"
    if not extracted_dir.exists():
        print(f"Directory not found: {extracted_dir}")
        return

    tex_stems = get_tex_stems(base)
    if not tex_stems:
        print("No .tex files found in base directory.")
        return

    xyz_files = sorted(extracted_dir.glob("*.xyz"))
    print(f"Found {len(xyz_files)} XYZ files. Adding comments...")

    for xyz_path in xyz_files:
        stem = xyz_path.stem
        paper_id, structure_name = parse_xyz_filename(stem, tex_stems)
        tex_path = base / f"{paper_id}.tex"
        title = extract_title_from_tex(tex_path)
        tex_filename = f"{paper_id}.tex"
        desc = structure_description(structure_name)

        comment = f"title: {title} | tex_file: {tex_filename} | structure: {desc}"
        add_comment_to_xyz(xyz_path, comment)
        print(f"  {xyz_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()
