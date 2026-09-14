# Build Vector Store for RAG Retrieval
# PowerShell script for building ChromaDB vector stores from authority_library packages
# Edit the parameters below to customize the build process

# Set error handling
$ErrorActionPreference = "Stop"

# ============ USER CONFIGURATION - Edit these parameters ============
# Package to build (required)
$package = "qiskit-aer"  # Options: "itensormps", "itensors", "netket", "qiskit", "qiskit-nature", "qiskit-algorithms", "qiskit-aer"

# ChromaDB storage path (relative to project root)
$chroma_path = "authority_library/rag_store/rag_chroma"

# Project root directory (leave empty for auto-detection)
$project_root = ""  # Example: "C:\path\to\project\root" (usually leave empty for auto-detection)

# ============ END USER CONFIGURATION ============

# Get script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$buildScript = Join-Path $scriptDir "build_vector_store.py"

# Build command arguments
$args_list = @(
    "--package", $package
)

if ($chroma_path) {
    $args_list += "--chroma-path"
    $args_list += $chroma_path
}

if ($project_root) {
    $args_list += "--project-root"
    $args_list += $project_root
}

# Change to script directory
Push-Location $scriptDir

try {
    $separator = "=" * 60
    Write-Host $separator
    Write-Host "Building vector store for package: $package"
    Write-Host $separator
    Write-Host ""
    
    # Run the build script
    python $buildScript $args_list
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host $separator
        Write-Host "Build completed successfully!"
        Write-Host $separator
    } else {
        Write-Host ""
        Write-Host "[ERROR] Build failed with exit code: $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
} catch {
    Write-Host ""
    Write-Host "[ERROR] Failed to build vector store: $_" -ForegroundColor Red
    exit 1
} finally {
    Pop-Location
}

