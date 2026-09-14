# MCP Retrieve Server

A Model Context Protocol (MCP) server providing RAG (Retrieval-Augmented Generation) retrieval services using hybrid search (vector similarity + BM25 keyword matching) with intelligent reranking.

## Overview

This MCP server enables AI assistants to perform semantic document retrieval from pre-built vector databases. It combines:

- **Vector Similarity Search**: Semantic search using sentence embeddings (BGE-base-en-v1.5)
- **BM25 Keyword Matching**: Traditional keyword-based retrieval
- **Hybrid Fusion**: Intelligent merging of both retrieval channels
- **Keyword Reranking**: Final reranking based on keyword matches and file type priority

The server supports multiple package collections (e.g., itensormps, itensors, netket, qiskit, qiskit-nature, qiskit-algorithms, qiskit-aer) and automatically manages ChromaDB vector stores and BM25 indices.

## Features

- `retrieve_hybrid`: Perform hybrid retrieval (vector + BM25) with reranking for a single package
- `retrieve_hybrid_multi`: Perform hybrid retrieval (vector + BM25) with reranking across multiple packages
- `list_available_packages`: List all available packages for retrieval
- **Automatic Model Preloading**: Embedding model is preloaded at server startup to avoid first-query delay
- **Automatic BM25 Index Rebuilding**: BM25 indices are automatically rebuilt from ChromaDB when needed
- **Configurable Parameters**: Adjustable weights for vector/BM25 fusion and reranking

## Installation

### Prerequisites

- Python 3.11 or higher
- ChromaDB vector store must be built before using the server (see "Building Vector Stores" below)

### Install the Package

```bash
cd MCP_servers/retrieve-mcp
pip install -e .
```

This will install the package and create the `mcp-server-retrieve` command.

### Install Dependencies

The package dependencies are listed in `requirements.txt`. If you need to install them separately:

```bash
pip install -r requirements.txt
```

## Pre-downloading the Embedding Model

The embedding model (BGE-base-en-v1.5, ~400MB) is **permanently stored in the project directory** (`retrieve-mcp/models/embedding_model/`). This ensures the model is available even after switching computers or clearing system caches.

### Quick Start (PowerShell - Recommended for Windows)

```powershell
cd MCP_servers/retrieve-mcp
.\download_embedding_model.ps1
```

### Alternative: Using Python Script

```bash
cd MCP_servers/retrieve-mcp
python download_embedding_model.py
```

### How It Works

1. **Model Storage**: The model is saved permanently to `retrieve-mcp/models/embedding_model/` (not in system cache).
2. **First Run**: If the model doesn't exist locally, it will be downloaded from Hugging Face (~400MB, may take a few minutes) and saved to the project directory.
3. **Subsequent Runs**: The model is loaded from the project directory (no download, but still takes a few seconds to load into memory).
4. **Fallback**: If the local model is missing or corrupted, the system will automatically download it again or fall back to Hugging Face cache.

**Model Storage Location**:
- **Project Directory**: `MCP_servers/retrieve-mcp/models/embedding_model/`
- The model is stored with the project, so it persists across:
  - Computer switches
  - System cache clearing
  - Reinstalling Python environments (as long as the project directory is preserved)

**Note**: 
- Even after downloading, loading the model into memory still takes a few seconds on each server startup. This is normal and expected behavior.
- The model directory (`models/embedding_model/`) can be added to `.gitignore` if you don't want to commit it to version control (it's ~400MB).
- If you want to share the model with team members, you can either commit it or have them run the download script.

## Testing

A lightweight test script is provided to quickly validate that the server components are working correctly:

```bash
cd MCP_servers/retrieve-mcp
python test_retrieve_mcp.py
```

The test script performs the following checks:
- **Module Imports**: Verifies all required modules can be imported
- **Configuration Loader**: Tests configuration file loading and parameter retrieval
- **Retrieval Module**: Validates retrieval module functions and package configurations
- **Server Parameter Logic**: Tests parameter handling logic (package vs numeric parameters)
- **Config File Structure**: Validates config.json structure and types

**Note**: This test script does NOT test actual retrieval functionality (which requires a built vector database). It only validates that the code components are properly structured and can load configurations correctly.

## Building Vector Stores

Before using the MCP retrieve server, you need to build vector stores for the packages you want to query.

### Quick Start (PowerShell - Recommended for Windows)

1. Edit `build_vector_store.ps1` and set the `$package` variable
2. Run the script:
   ```powershell
   cd MCP_servers/retrieve-mcp
   .\build_vector_store.ps1
   ```

### Alternative: Using Python Script

```bash
cd MCP_servers/retrieve-mcp
python build_vector_store.py --package itensormps
```

### Build Script Arguments

- `--package` (required): Package name to build (e.g. `itensormps`, `itensors`, `netket`)
- `--chroma-path` (optional): ChromaDB storage path relative to project root (default: `authority_library/rag_store/rag_chroma`)
- `--project-root` (optional): Project root directory (auto-detected if not provided)

### Example

```bash
# Build vector store for itensormps package
python build_vector_store.py --package itensormps

# Build for multiple packages
python build_vector_store.py --package itensormps
python build_vector_store.py --package itensors
python build_vector_store.py --package netket
```

**Note**: The vector store is stored in `authority_library/rag_store/rag_chroma` by default. This directory will be created automatically if it doesn't exist.

### Which libraries can be built

Package configs are defined in `src/mcp_server_retrieve/retrieval.py` (`get_package_configs()`). A package can be built only if its `doc_root` folder exists under `authority_library/`.

| Package (--package) | authority_library folder | Can build? |
|--------------------|---------------------------|------------|
| **orca-manual**    | `orca-manual/`           | Yes (.txt) |
| **qe-manual**      | `qe-manual/`             | Yes        |
| **qiskit**         | `qiskit/`                | Yes        |
| **qiskit-aer**     | `qiskit-aer/`            | Yes        |
| **qiskit-algorithms** | `qiskit-algorithms/`  | Yes        |
| **qiskit-nature**  | `qiskit-nature/`         | Yes        |
| **vasp-manual**    | `vasp-manual/`           | Yes        |
| **yamboo-manual**  | `yamboo-manual/`         | Yes        |
| **netket**         | `netket/`                | Yes        |
| **itensormps**     | `itensormps/`            | Only if you add this folder |
| **itensors**       | `itensors/`              | Only if you add this folder |

**Build one package:**
```bash
cd MCP_servers/retrieve-mcp
python build_vector_store.py --package orca-manual
```

**Build multiple packages (e.g. for DFT/ORCA workflows):**
```bash
python build_vector_store.py --package orca-manual
python build_vector_store.py --package qe-manual
python build_vector_store.py --package vasp-manual
```

**List configured packages:** Run with an invalid `--package` to see the error message listing available packages, or check `retrieval.py` → `get_package_configs()`.

### Customizing the Build Script

#### Using PowerShell Script (Recommended for Windows)

For Windows users, you can use the PowerShell script `build_vector_store.ps1` for easier parameter configuration:

```powershell
# Edit the parameters in build_vector_store.ps1, then run:
.\build_vector_store.ps1
```

Simply edit the variables at the top of `build_vector_store.ps1`:
- `$package`: Package name to build (e.g., "itensormps", "itensors", "netket")
- `$chroma_path`: ChromaDB storage path (default: "authority_library/rag_store/rag_chroma")
- `$project_root`: Project root directory (leave empty for auto-detection)

#### Using Python Script

The `build_vector_store.py` file is the user-facing entry point. You can edit this file directly to customize the build process (e.g., change default parameters, add custom logic). All core functionality is implemented in `src/mcp_server_retrieve/vector_store_builder.py`, which you typically don't need to modify.

### Configuring File Types

The file type configuration in `retrieval.py` **is automatically used** when building vector stores. Here's how it works:

**Configuration Flow**:
1. `build_vector_store.py` → calls `vector_store_builder.build_vector_store()`
2. `vector_store_builder.build_vector_store()` → calls `get_package_configs()` from `retrieval.py`
3. Extracts `file_extensions` from the config (line 428 in `vector_store_builder.py`)
4. Uses `file_extensions` to collect chunks (line 445 in `vector_store_builder.py`)

**So yes, the configuration in `retrieval.py` is used when building the vector store!**

**File Location**: `MCP_servers/retrieve-mcp/src/mcp_server_retrieve/retrieval.py`

**Function Location**: Lines 105-129 (the `get_package_configs()` function)

**How to Modify**:

1. Open `src/mcp_server_retrieve/retrieval.py`
2. Find the `get_package_configs()` function (around line 105)
3. Modify the `file_extensions` list for each package:

```python
def get_package_configs() -> Dict[str, Dict[str, Any]]:
    project_root = get_project_root()
    return {
        "itensormps": {
            "doc_root": project_root / "authority_library" / "itensormps",
            "collection_name": "itensor_mps_docs",
            "file_extensions": [".html", ".jl", ".py", ".md", ".txt"],  # ← Edit this list
        },
        "itensors": {
            "doc_root": project_root / "authority_library" / "itensors",
            "collection_name": "itensors_docs",
            "file_extensions": [".html", ".jl", ".py", ".md", ".txt"],  # ← Edit this list
        },
        "netket": {
            "doc_root": project_root / "authority_library" / "netket",
            "collection_name": "netket_docs",
            "file_extensions": [".html", ".jl", ".py", ".md", ".txt"],  # ← Edit this list
        },
    }
```

**Supported File Types**:
- `.html`: HTML documentation files
- `.py`, `.jl`, `.js`, `.ts`, `.cpp`, `.c`, `.h`, `.hpp`, `.java`, `.go`, `.rs`: Code files
- `.md`, `.markdown`: Markdown files
- `.txt`, `.text`: Plain text files

**Example**: To only include HTML and Markdown files for `itensormps`, change:
```python
"file_extensions": [".html", ".jl", ".py", ".md", ".txt"],
```
to:
```python
"file_extensions": [".html", ".md"],
```

**After modifying the configuration**, rebuild the vector store for the package to apply the changes:
```powershell
.\build_vector_store.ps1  # Make sure $package is set correctly
```

## Managing Vector Stores

This section describes tools for checking and managing existing vector stores.

### Checking Vector Store Statistics

The `check_vector_store` script allows you to view statistics about existing vector stores, including:
- Total number of chunks in each collection
- File type distribution (e.g., `.html`, `.py`, `.jl`, `.md`, `.txt`)
- Collection status (exists, empty, or not found)

#### Quick Start (PowerShell - Recommended for Windows)

1. Edit `check_vector_store.ps1` and configure the parameters:
   ```powershell
   $package = ""  # Leave empty to check all packages, or specify: "itensormps", "itensors", "netket"
   $chroma_path = "authority_library/rag_store/rag_chroma"
   $project_root = ""  # Leave empty for auto-detection
   $debug = $false  # Set to $true for detailed debug output
   ```

2. Run the script:
   ```powershell
   cd MCP_servers/retrieve-mcp
   .\check_vector_store.ps1
   ```

#### Alternative: Using Python Script

```bash
cd MCP_servers/retrieve-mcp
python check_vector_store.py
```

#### Script Arguments

- `--package` (optional): Package name to check (e.g. `itensormps`, `itensors`, `netket`). If not specified, checks all packages.
- `--chroma-path` (optional): ChromaDB storage path relative to project root (default: `authority_library/rag_store/rag_chroma`)
- `--project-root` (optional): Project root directory (auto-detected if not provided)
- `--debug` (optional): Enable detailed debug output

#### Example Output

```
======================================================================
Vector Store Statistics
======================================================================
Found 2 existing collection(s) in ChromaDB

======================================================================
Results
======================================================================

Package: itensormps
----------------------------------------------------------------------
  Collection name: itensor_mps_docs
  Chunks: 1,234
  File types (5 types):
    .html     :    500 chunks ( 40.5%)
    .jl       :    400 chunks ( 32.4%)
    .py       :    200 chunks ( 16.2%)
    .md       :     80 chunks (  6.5%)
    .txt      :     54 chunks (  4.4%)

Package: itensors
----------------------------------------------------------------------
  Collection name: itensors_docs
  Chunks: 2,456
  File types (4 types):
    .html     :  1,200 chunks ( 48.9%)
    .jl       :    800 chunks ( 32.6%)
    .py       :    300 chunks ( 12.2%)
    .md       :    156 chunks (  6.4%)

Package: netket
----------------------------------------------------------------------
  Collection name: netket_docs
  Status: Collection does not exist (not built yet)
  Chunks: 0
  File types: N/A

======================================================================
Total chunks across all packages: 3,690
======================================================================
```

#### Features

- **Read-only operation**: The script only reads from ChromaDB and never creates or modifies collections
- **Automatic collection detection**: Automatically detects which collections exist
- **File type analysis**: Extracts and displays file type distribution from metadata
- **Multiple counting methods**: Uses multiple fallback methods to ensure accurate chunk counting
- **Debug mode**: Use `--debug` flag for detailed diagnostic information

### Deleting Vector Store Collections

The `delete_collection` script allows you to safely delete a specific collection from ChromaDB. This is useful when you need to:
- Remove a partially built or corrupted collection
- Free up space before rebuilding a collection
- Clean up test collections

**⚠️ WARNING: This operation is permanent and cannot be undone!**

#### Quick Start (PowerShell - Recommended for Windows)

1. Edit `delete_collection.ps1` and configure the parameters:
   ```powershell
   $package = "netket"  # Package to delete: "itensormps", "itensors", or "netket"
   $chroma_path = "authority_library/rag_store/rag_chroma"
   $project_root = ""  # Leave empty for auto-detection
   $confirm = $false  # Set to $true to skip confirmation (dangerous!)
   ```

2. Run the script:
   ```powershell
   cd MCP_servers/retrieve-mcp
   .\delete_collection.ps1
   ```

3. If `$confirm = $false`, you will be prompted to confirm:
   ```
   Are you sure you want to delete this collection? (yes/no):
   ```
   Type `yes` or `y` to confirm deletion.

#### Alternative: Using Python Script

```bash
# With confirmation prompt (recommended)
python delete_collection.py --package netket

# Skip confirmation (dangerous!)
python delete_collection.py --package netket --confirm
```

#### Script Arguments

- `--package` (required): Package name to delete (e.g. `netket`, `itensors`, `itensormps`)
- `--chroma-path` (optional): ChromaDB storage path relative to project root (default: `authority_library/rag_store/rag_chroma`)
- `--project-root` (optional): Project root directory (auto-detected if not provided)
- `--confirm` (optional): Skip confirmation prompt (use with caution!)

#### Safety Features

- **Confirmation required by default**: The script requires explicit confirmation before deleting
- **Collection existence check**: Verifies the collection exists before attempting deletion
- **Read-only until confirmed**: The script only reads from ChromaDB until you confirm deletion
- **No accidental creation**: The script never creates new collections, even if the target collection doesn't exist

#### Example Output

```
======================================================================
Delete ChromaDB Collection
======================================================================

[INFO] ChromaDB path: D:\path\to\project\authority_library\rag_store\rag_chroma
[INFO] Package: netket
[INFO] Collection name: netket_docs
[INFO] Collection 'netket_docs' contains 384 chunks

======================================================================
WARNING: This will permanently delete the collection!
======================================================================
Package: netket
Collection: netket_docs
Chunks: 384

Are you sure you want to delete this collection? (yes/no): yes

[INFO] Deleting collection 'netket_docs'...
[SUCCESS] Collection 'netket_docs' deleted successfully!

======================================================================
Operation completed successfully!
======================================================================
```

#### If Collection Doesn't Exist

If the collection doesn't exist, the script will inform you without taking any action:

```
[INFO] Collection 'netket_docs' does not exist. Nothing to delete.
[INFO] Existing collections: ['itensor_mps_docs', 'itensors_docs']
[INFO] No action taken - database unchanged.
```

#### After Deletion

After deleting a collection, you can rebuild it using `build_vector_store.ps1`:

```powershell
# Edit build_vector_store.ps1 and set $package = "netket"
.\build_vector_store.ps1
```

## Running the Server

Once installed, you can use the `mcp-server-retrieve` command directly:

```bash
mcp-server-retrieve --chroma-path rag_store/rag_chroma --package itensormps --log-level INFO
```

### Command Line Arguments

- `--chroma-path`: (Optional) Path to ChromaDB storage directory (default: `authority_library/rag_store/rag_chroma`, relative to project root)
- `--package`: (Optional) Default package name (default: `itensormps`)
  - Available packages: `itensormps`, `itensors`, `netket`
- `--project-root`: (Optional) Path to project root directory (auto-detected if not provided)
  - The server automatically searches for the `authority_library` directory to find the project root
  - If auto-detection fails, provide this argument explicitly
- `--rt-target-dir`: (Optional) Target directory for saving retrieval results
  - If provided, retrieval results will be automatically saved to JSON files in this directory
  - Files are named based on the query text (sanitized for filesystem compatibility)
  - If not provided, retrieval results are returned to the LLM (default behavior for backward compatibility)
  - This is useful for performance optimization: saving results directly to files avoids passing large result dictionaries back to the LLM
- `--log-level`: (Optional) Set logging level (default: `INFO`)
  - Choices: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

## Configuration File

**IMPORTANT: This is the PRIMARY way to configure retrieval parameters.**

You **MUST** configure retrieval parameters (like `vector_num`, `bm25_num`, `final_num`, etc.) in the `config.json` file located in the `retrieve-mcp` root directory. This is the **only** place where you should set these parameters - do NOT modify the source code (e.g., `DEFAULT_CONFIG` in `config_loader.py`).

### Configuration File Location

**File**: `MCP_servers/retrieve-mcp/config.json`

**This is the ONLY file you need to edit to configure retrieval parameters.**

### Configuration Format

```json
{
  "retrieval": {
    "vector_num": 30,
    "bm25_num": 30,
    "final_num": 5,
    "vector_weight": 0.7,
    "reranker_weight": 0.3,
    "html_boost": 0.15
  },
  "default_package": "itensormps",
  "allow_parameter_override": true
}
```

### Configuration Parameters

- **`retrieval.vector_num`** (default: 30): Number of results from vector similarity search
- **`retrieval.bm25_num`** (default: 30): Number of results from BM25 keyword search
- **`retrieval.final_num`** (default: 5): Number of final results after reranking
- **`retrieval.vector_weight`** (default: 0.7): Weight for vector channel in fusion (0.0-1.0)
  - Higher value = more emphasis on semantic similarity
  - Lower value = more emphasis on keyword matching
- **`retrieval.reranker_weight`** (default: 0.3): Weight controlling keyword reranker influence (0.0-1.0)
  - Higher value = stronger keyword boost in final ranking
- **`retrieval.html_boost`** (default: 0.15): Additional boost for HTML files (0.0-1.0)
  - Higher value = stronger HTML priority
- **`default_package`** (default: "itensormps"): Default package name to use if not specified
- **`allow_parameter_override`** (default: true): Control whether **numeric parameters** passed to `retrieve_hybrid` can override config file values
  - **Note**: This setting only affects numeric parameters (vector_num, bm25_num, final_num, vector_weight, reranker_weight, html_boost)
  - **Note**: The `package` parameter is **not affected** by this setting and always prioritizes explicit parameter value
  - `true`: Numeric parameters can override config file values (default behavior)
  - `false`: Numeric parameters are ignored, only config file values are used (enforces consistent behavior)

### Configuration Priority and How It Works

**Configuration Priority (from highest to lowest):**
1. **`config.json` file** - This is where you should set your parameters
2. **`DEFAULT_CONFIG` in `config_loader.py`** - Only used as fallback if `config.json` doesn't exist or is missing a key
3. **Hardcoded defaults in code** - Only used if both above are unavailable

**How It Works:**

1. **When the server starts**, it automatically looks for `config.json` in the `retrieve-mcp` root directory
2. **If `config.json` exists**, the configuration is loaded and **values from `config.json` take precedence** over `DEFAULT_CONFIG`
3. **If `config.json` doesn't exist or is missing a key**, the system falls back to `DEFAULT_CONFIG` values
4. **Parameter Behavior**:
   - **`package` parameter**: Always prioritizes explicit parameter value. If not provided, uses default from server startup or config.json. This parameter is **not affected** by `allow_parameter_override`.
   - **Numeric parameters** (vector_num, bm25_num, final_num, vector_weight, reranker_weight, html_boost):
     - **Primary source**: Values from `config.json` (if the file exists and contains the key)
     - **Fallback**: `DEFAULT_CONFIG` values (only if `config.json` doesn't exist or is missing the key)
     - **If `allow_parameter_override` is `true` (default)**:
       - When calling `retrieve_hybrid`, any numeric parameters you explicitly provide will override the config file values
       - If a numeric parameter is not provided, it uses the value from `config.json`
     - **If `allow_parameter_override` is `false`**:
       - **All numeric parameters passed to `retrieve_hybrid` will be IGNORED**
       - Only values from `config.json` will be used
       - This ensures consistent behavior and prevents any parameter overrides

**Examples**:
- **Package parameter** (always uses explicit value if provided):
  - If you call `retrieve_hybrid` with `package: "itensors"`, it will use "itensors" regardless of config.json settings
  - If you don't provide `package`, it uses the default from server startup or config.json

- **Numeric parameters with `allow_parameter_override: true`**:
  - If you set `"vector_num": 50` in `config.json`, then `retrieve_hybrid` will use 50 by default
  - If you call with `vector_num: 100`, it will use 100 (overrides config file)
  - If you call without `vector_num`, it will use 50 from config file

- **Numeric parameters with `allow_parameter_override: false`**:
  - If you set `"vector_num": 50` in `config.json`, then `retrieve_hybrid` will **always** use 50
  - Even if you try to pass `vector_num: 100` in the function call, the 100 will be ignored

### Creating and Modifying the Configuration File

**To set retrieval parameters:**

1. **Edit `MCP_servers/retrieve-mcp/config.json`** - This is the ONLY file you need to modify
2. If the file doesn't exist, create it by copying the example format above
3. **Modify the values** in the `"retrieval"` section to your desired settings
4. **Save the file**
5. **Restart the MCP server** for changes to take effect (the configuration is cached at server startup)

**Example: To set `final_num` to 3:**
```json
{
  "retrieval": {
    "final_num": 3,
    ...
  }
}
```

**Important Notes:**
- **DO NOT modify** `DEFAULT_CONFIG` in `config_loader.py` - this is only a fallback
- **DO modify** `config.json` - this is the primary configuration file
- Changes to `config.json` require **restarting the MCP server** to take effect
- The configuration is loaded once at server startup and cached
- If you want to ensure your settings are always used, set `"allow_parameter_override": false` in `config.json`

## Configuration for mcp-use

To use this server with `mcp-use`, add it to your `MCP_server_config.json`:

```json
{
  "mcpServers": {
    "retrieve": {
      "command": "mcp-server-retrieve",
      "args": [
        "--chroma-path", "authority_library/rag_store/rag_chroma",
        "--package", "itensormps",
        "--log-level", "INFO"
      ]
    }
  }
}
```

**Note**: 
- **Path Resolution**: All relative paths (including `--chroma-path`) are resolved relative to the **project root directory**, not the current working directory. This means you can run the server from any directory (e.g., `run/` directory) and paths will still be correct.
- The `--project-root` argument is optional. The server will automatically detect the project root by searching for the `authority_library` directory, starting from the current working directory. If auto-detection fails, you can provide it explicitly:
  ```json
  "--project-root", "/path/to/project/root"
  ```
- **Example**: If you run the server from the `run/` directory, the default `rag_store/rag_chroma` path will be correctly resolved to `../rag_store/rag_chroma` (relative to project root), and `authority_library` paths will also be correctly resolved.

## Available Tools

### `retrieve_hybrid`

Perform hybrid retrieval (vector + BM25) with reranking for a single package.

**Behavior:**
- **If `--rt-target-dir` is set**: Retrieval results are automatically saved to JSON files in the target directory. The tool returns a minimal confirmation message instead of the full results, avoiding the overhead of passing large result dictionaries back to the LLM. This significantly improves performance.
- **If `--rt-target-dir` is not set**: Retrieval results are returned to the LLM as a dictionary (default behavior for backward compatibility).

**Parameters:**
- `query` (required): The search query text
- `package` (optional): Package name. If not provided, uses the default package configured at server startup or in config.json
- `vector_num` (optional): Number of results from vector similarity search
  - Default: from `config.json` `retrieval.vector_num` (30 if not configured)
- `bm25_num` (optional): Number of results from BM25 keyword search
  - Default: from `config.json` `retrieval.bm25_num` (30 if not configured)
- `final_num` (optional): Number of final results after reranking
  - Default: from `config.json` `retrieval.final_num` (5 if not configured)
- `vector_weight` (optional): Weight for vector channel in fusion (0.0-1.0)
  - Higher value = more emphasis on semantic similarity
  - Lower value = more emphasis on keyword matching
  - Default: from `config.json` `retrieval.vector_weight` (0.7 if not configured)
- `reranker_weight` (optional): Weight controlling keyword reranker influence (0.0-1.0)
  - Higher value = stronger keyword boost in final ranking
  - Default: from `config.json` `retrieval.reranker_weight` (0.3 if not configured)
- `html_boost` (optional): Additional boost for HTML files (0.0-1.0)
  - Higher value = stronger HTML priority
  - Default: from `config.json` `retrieval.html_boost` (0.15 if not configured)

**Note**: All parameters (except `query` and `package`) can be configured in `config.json`. If a parameter is not provided when calling the tool, it will use the value from `config.json`. If `config.json` doesn't exist or doesn't contain the parameter, hardcoded defaults are used.

**Returns:**

**If `--rt-target-dir` is set:**
- `status`: "success"
- `message`: Confirmation message with filename
- `query`: The original query
- `package`: Package name used
- `file_path`: Path to the saved JSON file
- `results_count`: Number of results saved
- `stats`: Statistics about the retrieval process

**If `--rt-target-dir` is not set (default):**
- `query`: The original query
- `package`: Package name used
- `results`: List of retrieval results with metadata
- `stats`: Statistics about the retrieval process

**Example Usage:**

```python
# In your MCP client code
result = await client.call_tool(
    "retrieve_hybrid",
    {
        "query": "How to calculate the energy of the excited state using DMRG?",
        "package": "itensormps",
        "vector_num": 30,
        "bm25_num": 30,
        "final_num": 5,
        "vector_weight": 0.7,
        "reranker_weight": 0.3,
        "html_boost": 0.15,
    }
)
```

### `retrieve_hybrid_multi`

Perform hybrid retrieval (vector + BM25) with reranking across multiple packages.

This tool performs hybrid retrieval for each specified package, then merges and reranks all results together. It combines vector similarity search and BM25 keyword matching for each package, merges results across packages with score re-normalization, and applies keyword-based reranking.

**Behavior:**
- **If `--rt-target-dir` is set**: Retrieval results are automatically saved to JSON files in the target directory. The tool returns a minimal confirmation message instead of the full results, avoiding the overhead of passing large result dictionaries back to the LLM. This significantly improves performance.
- **If `--rt-target-dir` is not set**: Retrieval results are returned to the LLM as a dictionary (default behavior for backward compatibility).

**Parameters:**
- `query` (required): The search query text
- `packages` (optional): List of package names (e.g. `["itensormps", "itensors", "netket"]`). If not provided, searches all available packages.
- `vector_num` (optional): Number of results from vector similarity search **per package**
  - Default: from `config.json` `retrieval.vector_num` (30 if not configured)
- `bm25_num` (optional): Number of results from BM25 keyword search **per package**
  - Default: from `config.json` `retrieval.bm25_num` (30 if not configured)
- `final_num` (optional): Number of final results after reranking **across all packages**
  - Default: from `config.json` `retrieval.final_num` (5 if not configured)
- `vector_weight` (optional): Weight for vector channel in fusion (0.0-1.0)
  - Higher value = more emphasis on semantic similarity
  - Lower value = more emphasis on keyword matching
  - Default: from `config.json` `retrieval.vector_weight` (0.7 if not configured)
- `reranker_weight` (optional): Weight controlling keyword reranker influence (0.0-1.0)
  - Higher value = stronger keyword boost in final ranking
  - Default: from `config.json` `retrieval.reranker_weight` (0.3 if not configured)
- `html_boost` (optional): Additional boost for HTML files (0.0-1.0)
  - Higher value = stronger HTML priority
  - Default: from `config.json` `retrieval.html_boost` (0.15 if not configured)

**Note**: All parameters (except `query` and `packages`) can be configured in `config.json`. The parameter override behavior (controlled by `allow_parameter_override` in config.json) applies the same way as `retrieve_hybrid`.

**Returns:**

**If `--rt-target-dir` is set:**
- `status`: "success"
- `message`: Confirmation message with filename
- `query`: The original query
- `packages`: List of package names searched
- `file_path`: Path to the saved JSON file
- `results_count`: Number of results saved
- `stats`: Statistics about the retrieval:
  - `total_vector_results_count`: Total number of vector results across all packages
  - `total_bm25_results_count`: Total number of BM25 results across all packages
  - `merged_results_count`: Number of merged results after combining packages
  - `final_results_count`: Number of final results after reranking
  - `per_package_stats`: Dictionary mapping package names to their individual stats

**If `--rt-target-dir` is not set (default):**
- `query`: The original query
- `packages`: List of package names searched
- `results`: List of retrieval results with metadata (same structure as `retrieve_hybrid`)
- `stats`: Statistics about the retrieval (same structure as above)

**Example Usage:**

```python
# In your MCP client code
result = await client.call_tool(
    "retrieve_hybrid_multi",
    {
        "query": "How to calculate the energy of the excited state using DMRG?",
        "packages": ["itensormps", "itensors"],
        "vector_num": 30,
        "bm25_num": 30,
        "final_num": 10,
        "vector_weight": 0.7,
        "reranker_weight": 0.3,
        "html_boost": 0.15,
    }
)

# Or search all available packages
result = await client.call_tool(
    "retrieve_hybrid_multi",
    {
        "query": "How to calculate the energy of the excited state using DMRG?",
        # packages not specified - will search all available packages
    }
)
```

**How Multi-Package Retrieval Works:**

1. **Per-Package Retrieval**: For each specified package, performs dual-channel retrieval (vector + BM25) and merges results using the same logic as `retrieve_hybrid`.

2. **Cross-Package Merging**: All results from all packages are collected and scores are re-normalized across all packages to ensure fair comparison. This is important because different packages may have different score distributions.

3. **Unified Reranking**: All merged results are reranked together using keyword matching and HTML boost, ensuring the best results across all packages are ranked highest.

4. **Result Formatting**: Top N results are returned, with metadata indicating which package each result came from.

### `list_available_packages`

List all available packages for retrieval.

**Returns:**
- `packages`: List of available package names
- `default_package`: Currently configured default package
- `package_info`: Dictionary mapping package names to their collection names

## How It Works

### Single-Package Retrieval (`retrieve_hybrid`)

1. **Model Preloading**: At server startup, the embedding model (BGE-base-en-v1.5) is preloaded into memory to avoid delay on first query.

2. **Dual-Channel Retrieval**:
   - **Vector Channel**: Uses ChromaDB to perform semantic similarity search with embeddings
   - **BM25 Channel**: Uses in-memory BM25 index for keyword matching (auto-rebuilt from ChromaDB if needed)

3. **Hybrid Fusion**: Results from both channels are merged with configurable weights, normalizing scores to a common scale [0, 1] in similarity form (higher is better).

4. **Keyword Reranking**: Final reranking based on:
   - Keyword match ratio (how many query keywords appear in the document)
   - File type priority (HTML files get additional boost)
   - All scores are adjusted in similarity space and results are sorted by similarity (highest first)

5. **Result Formatting**: Top N results are returned with comprehensive metadata including normalized scores (all in [0, 1] range, similarity form, higher is better), keyword matches, and source information.

### Multi-Package Retrieval (`retrieve_hybrid_multi`)

1. **Per-Package Retrieval**: For each specified package, performs the same dual-channel retrieval and hybrid fusion as single-package retrieval.

2. **Cross-Package Merging**: All results from all packages are collected and scores are **re-normalized across all packages** to ensure fair comparison. This is critical because different packages may have different score distributions.

3. **Unified Reranking**: All merged results are reranked together using keyword matching and HTML boost, ensuring the best results across all packages are ranked highest.

4. **Result Formatting**: Top N results are returned, with metadata indicating which package each result came from.

**Score Normalization**: All scores (`score`, `original_score`, `vector_score`, `bm25_score`) are normalized to [0, 1] range in similarity form, where higher values indicate better relevance. Results are sorted in descending order by score (most similar first).

**Important: Relative Normalization Within Each Query Batch**

The normalization is performed **relative to each query's retrieval batch**, not globally across all queries. This means:

1. **Vector Score Normalization**:
   - For each query, vector scores are normalized based on the min/max distances within that query's `vector_num` results
   - The result with the smallest distance (most similar) in the batch gets `vector_score = 1.0`
   - This is why you may see multiple results with `vector_score = 1.0` - they are the best matches within their respective query batches

2. **BM25 Score Normalization**:
   - For each query, BM25 scores are normalized based on the min/max scores within that query's `bm25_num` results
   - The result with the highest BM25 score in the batch gets `bm25_score = 1.0`
   - Similarly, multiple results may have `bm25_score = 1.0` if they are the best matches in their batches

**Why This Design?**
- **Advantage**: Scores are comparable within each query's results, making it easy to identify the best matches for that specific query
- **Note**: Scores from different queries are not directly comparable, as each query has its own normalization baseline
- **Common Pattern**: You will often see `vector_score = 1.0` or `bm25_score = 1.0` for the top results in each query, which is expected behavior

This relative normalization ensures that each query's results are properly scaled for fusion and reranking, regardless of the absolute score ranges from the underlying retrieval systems.

## Architecture

```
retrieve-mcp/
├── src/
│   └── mcp_server_retrieve/
│       ├── __init__.py
│       ├── retrieval.py           # Core retrieval functions (extracted from rag_demo.py)
│       ├── vector_store_builder.py # Vector store building functions
│       ├── server.py              # FastMCP server and tool definitions
│       ├── main.py                # Entry point and CLI
│       └── config_loader.py       # Configuration file loader
├── models/
│   └── embedding_model/           # Embedding model storage (created after running download script)
├── build_vector_store.py          # User-facing entry point for building vector stores
├── build_vector_store.ps1         # PowerShell script for building vector stores
├── check_vector_store.py          # Check vector store statistics (chunks, file types)
├── check_vector_store.ps1         # PowerShell script for checking vector stores
├── delete_collection.py            # Delete a ChromaDB collection
├── delete_collection.ps1          # PowerShell script for deleting collections
├── download_embedding_model.py    # Pre-download embedding model (Python)
├── download_embedding_model.ps1   # Pre-download embedding model (PowerShell)
├── test_retrieve_mcp.py          # Lightweight test script
├── rag_demo.py                    # Original demo script (not modified)
├── config.json                    # Configuration file for retrieval parameters
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Package configuration
└── README.md                      # This file
```

## Dependencies

- `mcp`: Model Context Protocol framework
- `chromadb`: Vector database for storing embeddings
- `sentence-transformers`: Embedding model (BGE-base-en-v1.5)
- `rank-bm25`: BM25 keyword matching implementation

## Notes

- The server uses FastMCP for operation
- **ChromaDB Location**: Vector stores are stored in `authority_library/rag_store/rag_chroma` by default. This path is relative to the project root directory.
- **Project root auto-detection**: The server automatically finds the project root by searching for the `authority_library` directory, starting from the current working directory or the server's file location. This ensures that `authority_library` paths are correctly resolved regardless of where the server is started from.
- **Path Resolution**: All relative paths (including `--chroma-path`) are resolved relative to the **project root directory**, not the current working directory. This means you can run the server from any directory (e.g., `run/` directory) and paths will still be correct.
- BM25 indices are built in-memory and automatically rebuilt from ChromaDB when needed
- The embedding model is cached in memory after first load
- All file operations and vector stores are read-only (no modifications to ChromaDB)

## Troubleshooting

### Model Loading Issues

If you encounter issues loading the embedding model:
- Ensure you have sufficient disk space (model is ~400MB)
- Check your internet connection (model downloads from Hugging Face on first use if not in project directory)
- Verify `sentence-transformers` is properly installed
- **Pre-download the model**: Run `download_embedding_model.py` or `download_embedding_model.ps1` to download the model to the project directory (`models/embedding_model/`)
- **Model location**: The model is stored in `retrieve-mcp/models/embedding_model/`. If you have issues, you can delete this directory and re-run the download script
- **Fallback behavior**: If the local model is missing, the system will automatically try to download it or fall back to Hugging Face cache

### ChromaDB Not Found

If the server cannot find ChromaDB:
- Verify the `--chroma-path` points to the correct directory
- Ensure the ChromaDB was built using `rag_demo.py` or similar
- Check that the collection names match the package configuration

### BM25 Index Rebuilding

If BM25 index rebuilding is slow:
- This is normal on first use or after ChromaDB updates
- The index is cached in memory for subsequent queries
- Consider pre-building indices if you have many packages

## License

MIT License

