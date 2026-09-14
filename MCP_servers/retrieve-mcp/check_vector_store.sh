#!/bin/bash
# Check Vector Store Statistics
# Bash script for checking chunks count and file types in existing vector stores
# Edit the parameters below to customize the check process

# Set error handling
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ============ USER CONFIGURATION - Edit these parameters ============
# Package to check (leave empty to check all packages)
package=""  # Options: "itensormps", "itensors", "netket", "qiskit", "qiskit-nature", "qiskit-algorithms", "qiskit-aer", or "" for all

# ChromaDB storage path (relative to project root)
chroma_path="authority_library/rag_store/rag_chroma"

# Project root directory (leave empty for auto-detection)
project_root=""  # Example: "/path/to/project/root" (usually leave empty for auto-detection)

# Enable debug output (detailed information)
debug=false  # Set to true to see detailed debug information

# ============ END USER CONFIGURATION ============

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK_SCRIPT="$SCRIPT_DIR/check_vector_store.py"

# Build command arguments
args_list=()

if [ -n "$package" ]; then
    args_list+=("--package")
    args_list+=("$package")
fi

if [ -n "$chroma_path" ]; then
    args_list+=("--chroma-path")
    args_list+=("$chroma_path")
fi

if [ -n "$project_root" ]; then
    args_list+=("--project-root")
    args_list+=("$project_root")
fi

if [ "$debug" = true ]; then
    args_list+=("--debug")
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

# Create separator (70 equal signs)
separator="======================================================================"

echo -e "${CYAN}$separator${NC}"
echo -e "${CYAN}Checking vector store statistics${NC}"
if [ -n "$package" ]; then
    echo -e "${CYAN}Package: $package${NC}"
else
    echo -e "${CYAN}Packages: All${NC}"
fi
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

# Run the check script (disable errexit temporarily to check exit code)
set +e
$PYTHON_CMD "$CHECK_SCRIPT" "${args_list[@]}"
exit_code=$?
set -e

if [ $exit_code -eq 0 ]; then
    echo ""
    echo -e "${GREEN}$separator${NC}"
    echo -e "${GREEN}Check completed successfully!${NC}"
    echo -e "${GREEN}$separator${NC}"
    trap - EXIT  # Remove trap on success
    exit 0
else
    echo ""
    echo -e "${RED}[ERROR] Check failed with exit code: $exit_code${NC}"
    exit $exit_code
fi

