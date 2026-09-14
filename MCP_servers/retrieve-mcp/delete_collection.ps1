# Delete ChromaDB Collection
# PowerShell script for deleting a specific collection from ChromaDB
# Use with caution - this operation cannot be undone!

# Set error handling
$ErrorActionPreference = "Stop"

# ============ USER CONFIGURATION - Edit these parameters ============
# Package to delete (required)
$package = "netket"  # Options: "itensormps", "itensors", "netket", "qiskit", "qiskit-nature", "qiskit-algorithms", "qiskit-aer"

# ChromaDB storage path (relative to project root)
$chroma_path = "authority_library/rag_store/rag_chroma"

# Project root directory (leave empty for auto-detection)
$project_root = ""  # Example: "C:\path\to\project\root" (usually leave empty for auto-detection)

# Skip confirmation prompt (use with caution!)
# Set to $true to delete without confirmation, $false to show confirmation
$confirm = $false  # WARNING: Setting to $true will delete without asking!

# ============ END USER CONFIGURATION ============

# Get script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$deleteScript = Join-Path $scriptDir "delete_collection.py"

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

if ($confirm) {
    $args_list += "--confirm"
}

# Change to script directory
Push-Location $scriptDir

try {
    $separator = "=" * 70
    Write-Host $separator
    Write-Host "Deleting ChromaDB Collection"
    Write-Host "Package: $package"
    Write-Host $separator
    Write-Host ""
    
    # Run the delete script
    python $deleteScript $args_list
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host $separator
        Write-Host "Operation completed successfully!"
        Write-Host $separator
    } else {
        Write-Host ""
        Write-Host "[ERROR] Operation failed with exit code: $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
} catch {
    Write-Host ""
    Write-Host "[ERROR] Failed to delete collection: $_" -ForegroundColor Red
    exit 1
} finally {
    Pop-Location
}

