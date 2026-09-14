#!/bin/bash
# Build Vector Store for RAG Retrieval
# Bash script for building ChromaDB vector stores from authority_library packages
# Edit the parameters below to customize the build process

# Set error handling
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ============ USER CONFIGURATION - Edit these parameters ============
# Package to build (required)
package="qiskit-aer"  # Options: "itensormps", "itensors", "netket", "qiskit", "qiskit-nature", "qiskit-algorithms", "qiskit-aer"

# ChromaDB storage path (relative to project root)
chroma_path="authority_library/rag_store/rag_chroma"

# Project root directory (leave empty for auto-detection)
project_root=""  # Example: "/path/to/project/root" (usually leave empty for auto-detection)

# ============ END USER CONFIGURATION ============

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_SCRIPT="$SCRIPT_DIR/build_vector_store.py"

# Build command arguments
args_list=(
    "--package" "$package"
)

if [ -n "$chroma_path" ]; then
    args_list+=("--chroma-path")
    args_list+=("$chroma_path")
fi

if [ -n "$project_root" ]; then
    args_list+=("--project-root")
    args_list+=("$project_root")
fi

# Change to script directory
cd "$SCRIPT_DIR"

# Function to handle errors
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ] && [ $exit_code -ne 1 ]; then
        echo ""
        echo -e "${RED}[ERROR] Unexpected error occurred${NC}"
    fi
}

# Set trap to call cleanup on exit
trap cleanup EXIT

# Create separator (60 equal signs)
separator="============================================================"

echo -e "${CYAN}$separator${NC}"
echo -e "${CYAN}Building vector store for package: $package${NC}"
echo -e "${CYAN}$separator${NC}"
echo ""

# Check if Python is available
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}[ERROR] Python is not installed or not in PATH.${NC}"
    exit 1
fi

# Run the build script (disable errexit temporarily to check exit code)
set +e
$PYTHON_CMD "$BUILD_SCRIPT" "${args_list[@]}"
exit_code=$?
set -e

if [ $exit_code -eq 0 ]; then
    echo ""
    echo -e "${GREEN}$separator${NC}"
    echo -e "${GREEN}Build completed successfully!${NC}"
    echo -e "${GREEN}$separator${NC}"
    trap - EXIT  # Remove trap on success
    exit 0
else
    echo ""
    echo -e "${RED}[ERROR] Build failed with exit code: $exit_code${NC}"
    exit $exit_code
fi

