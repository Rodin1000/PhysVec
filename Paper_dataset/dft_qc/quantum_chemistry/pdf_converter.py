import fitz  # PyMuPDF
from pathlib import Path

def pdf_to_text_fitz(pdf_path: str, out_txt: str | None = None) -> str:
    """Convert a PDF file to text."""
    doc = fitz.open(pdf_path)
    parts = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text")  # plain text
        parts.append(f"\n\n--- Page {i} ---\n{text}")
    doc.close()

    full_text = "".join(parts).strip()
    if out_txt:
        Path(out_txt).write_text(full_text, encoding="utf-8")
    return full_text


def convert_all_pdfs_to_text(folder_path: str | Path = ".") -> None:
    """Convert all PDF files in the specified folder to text files."""
    folder = Path(folder_path)
    pdf_files = list(folder.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {folder.absolute()}")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s) to convert...")
    
    for pdf_file in pdf_files:
        try:
            # Create output filename by replacing .pdf with .txt
            output_file = pdf_file.with_suffix(".txt")
            
            print(f"Converting: {pdf_file.name} -> {output_file.name}")
            pdf_to_text_fitz(str(pdf_file), str(output_file))
            print(f"  ✓ Successfully converted {pdf_file.name}")
            
        except Exception as e:
            print(f"  ✗ Error converting {pdf_file.name}: {e}")
    
    print(f"\nConversion complete! Converted {len(pdf_files)} PDF file(s).")


if __name__ == "__main__":
    # Convert all PDFs in the current folder
    convert_all_pdfs_to_text(".")
