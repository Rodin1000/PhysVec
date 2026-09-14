#!/bin/bash
# Bash script to pre-download the embedding model for retrieve-mcp
#
# This script downloads the BGE-base-en-v1.5 embedding model from Hugging Face
# and saves it permanently to the project directory (retrieve-mcp/models/embedding_model/).
#
# This ensures the model is stored with the project and doesn't need to be
# re-downloaded when switching computers or clearing system caches.
#
# Usage:
#   ./download_embedding_model.sh

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}Downloading Embedding Model for retrieve-mcp${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if Python is available
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}[ERROR] Python is not installed or not in PATH.${NC}"
    echo -e "${RED}Please install Python 3.11 or higher and try again.${NC}"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo -e "${GREEN}[INFO] Python found: $PYTHON_VERSION${NC}"

# Check if sentence-transformers is installed
echo -e "${YELLOW}[INFO] Checking if sentence-transformers is installed...${NC}"
if $PYTHON_CMD -c "import sentence_transformers; print('OK')" 2>&1 > /dev/null; then
    echo -e "${GREEN}[INFO] sentence-transformers is already installed.${NC}"
else
    echo -e "${YELLOW}[WARNING] sentence-transformers is not installed.${NC}"
    echo -e "${YELLOW}[INFO] Installing sentence-transformers...${NC}"
    $PYTHON_CMD -m pip install sentence-transformers
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR] Failed to install sentence-transformers.${NC}"
        exit 1
    fi
fi

# Run the download script
echo ""
echo -e "${YELLOW}[INFO] Running download script...${NC}"
echo ""

$PYTHON_CMD download_embedding_model.py

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}Download completed successfully!${NC}"
    echo -e "${GREEN}============================================================${NC}"
    echo ""
    echo -e "${GREEN}Model saved to: models/embedding_model/${NC}"
    echo -e "${GREEN}The model is now permanently stored in the project directory.${NC}"
    echo ""
    echo -e "${GREEN}You can now use retrieve-mcp without download delays.${NC}"
    echo -e "${YELLOW}Note: The model still needs to be loaded into memory on first use,${NC}"
    echo -e "${YELLOW}      which takes a few seconds, but there will be no download delay.${NC}"
else
    echo ""
    echo -e "${RED}[ERROR] Download failed. Please check the error messages above.${NC}"
    exit 1
fi

