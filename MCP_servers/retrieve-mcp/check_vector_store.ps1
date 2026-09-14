# Check Vector Store Statistics
# PowerShell script for checking chunks count and file types in existing vector stores
# Edit the parameters below to customize the check process

# Set error handling
$ErrorActionPreference = "Stop"

# ============ USER CONFIGURATION - Edit these parameters ============
# Package to check (leave empty to check all packages)
$package = ""  # Options: "itensormps", "itensors", "netket", "qiskit", "qiskit-nature", "qiskit-algorithms", "qiskit-aer", or "" for all

# ChromaDB storage path (relative to project root)
$chroma_path = "authority_library/rag_store/rag_chroma"

# Project root directory (leave empty for auto-detection)
$project_root = ""  # Example: "C:\path\to\project\root" (usually leave empty for auto-detection)

# Enable debug output (detailed information)
$debug = $false  # Set to $true to see detailed debug information

# ============ END USER CONFIGURATION ============

# Get script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$checkScript = Join-Path $scriptDir "check_vector_store.py"

# Build command arguments
$args_list = @()

if ($package) {
    $args_list += "--package"
    $args_list += $package
}

if ($chroma_path) {
    $args_list += "--chroma-path"
    $args_list += $chroma_path
}

if ($project_root) {
    $args_list += "--project-root"
    $args_list += $project_root
}

if ($debug) {
    $args_list += "--debug"
}

# Change to script directory
Push-Location $scriptDir

try {
    $separator = "=" * 70
    Write-Host $separator
    Write-Host "Checking vector store statistics"
    if ($package) {
        Write-Host "Package: $package"
    } else {
        Write-Host "Packages: All"
    }
    Write-Host $separator
    Write-Host ""
    
    # Run the check script
    python $checkScript $args_list
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host $separator
        Write-Host "Check completed successfully!"
        Write-Host $separator
    } else {
        Write-Host ""
        Write-Host "[ERROR] Check failed with exit code: $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
} catch {
    Write-Host ""
    Write-Host "[ERROR] Failed to check vector store: $_" -ForegroundColor Red
    exit 1
} finally {
    Pop-Location
}

