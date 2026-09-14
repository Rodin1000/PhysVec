# PowerShell script to pre-download the embedding model for retrieve-mcp
#
# This script downloads the BGE-base-en-v1.5 embedding model from Hugging Face
# and saves it permanently to the project directory (retrieve-mcp/models/embedding_model/).
#
# This ensures the model is stored with the project and doesn't need to be
# re-downloaded when switching computers or clearing system caches.
#
# Usage:
#   .\download_embedding_model.ps1

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Downloading Embedding Model for retrieve-mcp" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Get the script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[INFO] Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.11 or higher and try again." -ForegroundColor Red
    exit 1
}

# Check if sentence-transformers is installed
Write-Host "[INFO] Checking if sentence-transformers is installed..." -ForegroundColor Yellow
$checkResult = python -c "import sentence_transformers; print('OK')" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARNING] sentence-transformers is not installed." -ForegroundColor Yellow
    Write-Host "[INFO] Installing sentence-transformers..." -ForegroundColor Yellow
    pip install sentence-transformers
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to install sentence-transformers." -ForegroundColor Red
        exit 1
    }
}

# Run the download script
Write-Host ""
Write-Host "[INFO] Running download script..." -ForegroundColor Yellow
Write-Host ""

python download_embedding_model.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "Download completed successfully!" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Model saved to: models\embedding_model\" -ForegroundColor Green
    Write-Host "The model is now permanently stored in the project directory." -ForegroundColor Green
    Write-Host ""
    Write-Host "You can now use retrieve-mcp without download delays." -ForegroundColor Green
    Write-Host "Note: The model still needs to be loaded into memory on first use," -ForegroundColor Yellow
    Write-Host "      which takes a few seconds, but there will be no download delay." -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "[ERROR] Download failed. Please check the error messages above." -ForegroundColor Red
    exit 1
}

