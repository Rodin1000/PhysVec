#!/usr/bin/env python3
"""
Pre-download the embedding model for retrieve-mcp.

This script downloads the BGE-base-en-v1.5 embedding model from Hugging Face
and saves it permanently to the project directory (retrieve-mcp/models/embedding_model/).

This ensures the model is stored with the project and doesn't need to be
re-downloaded when switching computers or clearing system caches.

After running this script, the model will be available in the project directory
and subsequent runs of retrieve-mcp will load it from there (still takes a few
seconds to load into memory, but no download time).

Usage:
    python download_embedding_model.py
"""

import sys
import os
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers is not installed.")
    print("Please install it with: pip install sentence-transformers")
    sys.exit(1)

# Model name (must match the one used in retrieval.py)
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Get the script directory (retrieve-mcp root)
script_dir = Path(__file__).parent.resolve()
MODEL_DIR = script_dir / "models" / "embedding_model"


def main():
    print("=" * 70)
    print("Downloading Embedding Model for retrieve-mcp")
    print("=" * 70)
    print(f"\nModel: {EMBED_MODEL_NAME}")
    print(f"Target directory: {MODEL_DIR}")
    print("\nThis may take a few minutes depending on your internet connection.")
    print("The model is approximately 400MB.")
    print("\nThe model will be saved permanently to the project directory.")
    print("This ensures it's available even after clearing system caches.")
    print("=" * 70)
    print()

    # Check if model already exists
    if MODEL_DIR.exists() and (MODEL_DIR / "config.json").exists():
        print(f"[INFO] Model already exists at: {MODEL_DIR}")
        response = input("Do you want to re-download it? (y/N): ").strip().lower()
        if response != 'y':
            print("[INFO] Skipping download. Using existing model.")
            return 0
        print("[INFO] Re-downloading model...")
        print()

    try:
        print(f"[INFO] Downloading model: {EMBED_MODEL_NAME} from Hugging Face...")
        print("[INFO] This may take a few minutes...")
        print()

        # Download the model (this will use Hugging Face cache if available)
        model = SentenceTransformer(EMBED_MODEL_NAME)

        # Create the model directory
        MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)

        # Save the model to the project directory
        print(f"\n[INFO] Saving model to: {MODEL_DIR}...")
        model.save(str(MODEL_DIR))

        print()
        print("=" * 70)
        print("SUCCESS: Model downloaded and saved successfully!")
        print("=" * 70)
        print(f"\nModel location: {MODEL_DIR}")
        print(f"Size: ~400MB")
        print(f"\nSubsequent runs of retrieve-mcp will load the model from this location.")
        print(f"No need to re-download when switching computers or clearing caches.")
        print(f"\nNote: Loading the model into memory still takes a few seconds,")
        print(f"      but there will be no download delay.")
        print()

        return 0

    except KeyboardInterrupt:
        print("\n\n[ERROR] Download interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Failed to download model: {e}")
        print("\nPossible causes:")
        print("  - Network connection issues")
        print("  - Insufficient disk space (model requires ~400MB)")
        print("  - Hugging Face API issues")
        print("  - Permission issues (cannot write to project directory)")
        print("\nPlease check your internet connection and try again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

