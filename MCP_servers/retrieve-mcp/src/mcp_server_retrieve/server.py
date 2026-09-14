"""MCP Retrieve Server - FastMCP server implementation."""

import logging
import json
from typing import Any, Dict, List, Optional, Union

from mcp.server.fastmcp import FastMCP

from mcp_server_retrieve.retrieval import (
    preload_embed_model,
    retrieve_vector,
    retrieve_bm25,
    merge_retrieval_results,
    merge_multi_package_results,
    keyword_reranker,
    DEFAULT_PACKAGE,
    get_package_configs,
    set_project_root,
    find_project_root,
)
from mcp_server_retrieve.config_loader import (
    get_retrieval_config,
    get_default_package,
    get_allow_parameter_override,
    load_config,
)

# Initialize logger
logger = logging.getLogger(__name__)

# Create a FastMCP server instance
mcp = FastMCP("Retrieve Service")

# Store configuration
_chroma_path: Optional[str] = None
_default_package: str = DEFAULT_PACKAGE
_retrieval_config: Optional[Dict[str, Any]] = None
_rt_target_dir: Optional[str] = None  # Target directory for saving retrieval results


def set_chroma_path(chroma_path: str) -> None:
    """Set the ChromaDB storage path.
    
    Args:
        chroma_path: Path to ChromaDB storage directory
    """
    global _chroma_path
    _chroma_path = chroma_path
    logger.info("ChromaDB path set to: %s", _chroma_path)


def set_default_package(package: str) -> None:
    """Set the default package name.
    
    Args:
        package: Package name (e.g. "itensormps", "itensors", "netket")
    """
    global _default_package
    configs = get_package_configs()
    if package not in configs:
        raise ValueError(
            f"Unknown package name: {package}. "
            f"Available packages: {list(configs.keys())}"
        )
    _default_package = package
    logger.info("Default package set to: %s", _default_package)


def set_rt_target_dir(rt_target_dir: str) -> None:
    """Set the target directory for saving retrieval results.
    
    Args:
        rt_target_dir: Path to directory where retrieval results will be saved
    """
    global _rt_target_dir
    from pathlib import Path
    _rt_target_dir = str(Path(rt_target_dir).expanduser().resolve())
    logger.info("Retrieval target directory set to: %s", _rt_target_dir)


def _get_retrieval_config() -> Dict[str, Any]:
    """Get retrieval configuration, loading from file if not already loaded."""
    global _retrieval_config
    if _retrieval_config is None:
        _retrieval_config = get_retrieval_config()
    return _retrieval_config


@mcp.tool()
def retrieve_hybrid(
    query: str,
    package: str,
) -> Dict[str, Any]:
    """
    Perform hybrid retrieval (vector + BM25) with reranking for a single package.
    
    This tool combines vector similarity search and BM25 keyword matching, then
    applies keyword-based reranking to return the most relevant results.
    
    **Behavior:**
    - If the server was started with `--rt-target-dir`, retrieval results are automatically
      saved to JSON files in that directory. The tool returns a minimal confirmation message
      instead of the full results, significantly improving performance by avoiding the overhead
      of passing large result dictionaries back to the LLM.
    - If `--rt-target-dir` was not set, the tool returns the full results as a dictionary
      (default behavior for backward compatibility).
    
    The model selects a package explicitly. If package is an empty string, the server
    startup/default package is used. Retrieval counts and weights are loaded
    from config.json and are not exposed as tool inputs.
    
    Args:
        query: The search query text
        package: Package name (e.g. "itensormps", "itensors", "netket"). An empty
                 value uses the package selected by the server startup/default configuration.
    
    Returns:
        **If `--rt-target-dir` is set (automatic file saving mode):**
        Dictionary containing:
        - status: "success"
        - message: Confirmation message with filename
        - query: The original query
        - package: Package name used
        - file_path: Path to the saved JSON file
        - results_count: Number of results saved
        - stats: Statistics about the retrieval:
            - vector_results_count: Number of vector results
            - bm25_results_count: Number of BM25 results
            - merged_results_count: Number of merged results
            - final_results_count: Number of final results after reranking
        
        **If `--rt-target-dir` is not set (default mode):**
        Dictionary containing:
        - query: The original query
        - package: Package name used
        - results: List of retrieval results, each containing:
            - id: Chunk ID
            - text: Document text
            - metadata: Metadata (source_file, section_title, section_id, package)
            - score: Final reranked score (similarity form, higher is better, normalized to [0, 1])
            - original_score: Original retrieval score (similarity form, higher is better, normalized to [0, 1])
            - vector_score: Normalized vector similarity score (similarity form, higher is better, normalized to [0, 1], if available)
            - bm25_score: Normalized BM25 score (similarity form, higher is better, normalized to [0, 1], if available)
            
            **Note**: All scores are normalized to [0, 1] range in similarity form (higher is better).
            Results are sorted by score in descending order (highest similarity first).
            - keyword_matches: Number of matched keywords
            - matched_keywords: List of matched keywords
            - match_ratio: Ratio of matched keywords
            - is_html: Whether source file is HTML
        - stats: Statistics about the retrieval:
            - vector_results_count: Number of vector results
            - bm25_results_count: Number of BM25 results
            - merged_results_count: Number of merged results
            - final_results_count: Number of final results after reranking
    """
    vector_num = None
    bm25_num = None
    final_num = None
    vector_weight = None
    reranker_weight = None
    html_boost = None

    try:
        # Load configuration from file (if not already loaded)
        # _get_retrieval_config() returns the "retrieval" section from config.json (or DEFAULT_CONFIG if not found)
        config = _get_retrieval_config()
        allow_override = get_allow_parameter_override()
        
        # Log loaded configuration for debugging
        logger.debug("Loaded retrieval config: %s", config)
        
        # Package parameter: Always prioritize explicit parameter (not affected by allow_parameter_override)
        if not package:
            package = _default_package
        
        # Numeric parameters: Use values from config.json (loaded via _get_retrieval_config())
        # _get_retrieval_config() returns the "retrieval" section from config.json,
        # or DEFAULT_CONFIG["retrieval"] if config.json doesn't exist.
        # The config dictionary already contains the merged values (config.json values override DEFAULT_CONFIG).
        
        # Extract values from config (these come from config.json if present, otherwise from DEFAULT_CONFIG)
        # _get_retrieval_config() already merges config.json with DEFAULT_CONFIG, so these values
        # are guaranteed to exist (either from config.json or from DEFAULT_CONFIG as fallback)
        config_vector_num = config.get("vector_num")
        config_bm25_num = config.get("bm25_num")
        config_final_num = config.get("final_num")
        config_vector_weight = config.get("vector_weight")
        config_reranker_weight = config.get("reranker_weight")
        config_html_boost = config.get("html_boost")
        
        # Log the values being used (for debugging)
        logger.debug("Config values from config.json: final_num=%s, vector_num=%s, bm25_num=%s", 
                     config_final_num, config_vector_num, config_bm25_num)
        
        if not allow_override:
            # Force use config file values, ignore any provided numeric parameters
            logger.info("Parameter override disabled in config.json. Using config file values for numeric parameters (ignoring any provided parameters).")
            vector_num = config_vector_num
            bm25_num = config_bm25_num
            final_num = config_final_num
            vector_weight = config_vector_weight
            reranker_weight = config_reranker_weight
            html_boost = config_html_boost
        else:
            # Use config file values as defaults, but allow override if explicitly provided
            vector_num = config_vector_num if vector_num is None else vector_num
            bm25_num = config_bm25_num if bm25_num is None else bm25_num
            final_num = config_final_num if final_num is None else final_num
            vector_weight = config_vector_weight if vector_weight is None else vector_weight
            reranker_weight = config_reranker_weight if reranker_weight is None else reranker_weight
            html_boost = config_html_boost if html_boost is None else html_boost
            
            # Apply explicit parameter overrides if provided
            if vector_num is not None:
                vector_num = vector_num
            if bm25_num is not None:
                bm25_num = bm25_num
            if final_num is not None:
                final_num = final_num
            if vector_weight is not None:
                vector_weight = vector_weight
            if reranker_weight is not None:
                reranker_weight = reranker_weight
            if html_boost is not None:
                html_boost = html_boost
        
        configs = get_package_configs()
        if package not in configs:
            raise ValueError(
                f"Unknown package name: {package}. "
                f"Available packages: {list(configs.keys())}"
            )
        
        logger.info("Performing hybrid retrieval for package '%s' with query: %s", package, query[:100])
        logger.debug("Using parameters: vector_num=%d, bm25_num=%d, final_num=%d, vector_weight=%.2f, reranker_weight=%.2f, html_boost=%.2f",
                     vector_num, bm25_num, final_num, vector_weight, reranker_weight, html_boost)
        
        # Step 1: Dual-channel retrieval
        vector_results = retrieve_vector(query, k=vector_num, package=package, chroma_path=_chroma_path)
        logger.debug("Vector channel: %d results", len(vector_results))
        
        bm25_results = retrieve_bm25(query, k=bm25_num, package=package, chroma_path=_chroma_path)
        logger.debug("BM25 channel: %d results", len(bm25_results))
        
        # Step 2: Merge results from both channels
        merged_results = merge_retrieval_results(
            vector_results,
            bm25_results,
            vector_weight=vector_weight
        )
        logger.debug("Merged results: %d unique chunks", len(merged_results))
        
        # Step 3: Keyword reranking with HTML priority
        reranked_results = keyword_reranker(
            merged_results,
            query,
            reranker_weight=reranker_weight,
            html_boost=html_boost
        )
        
        # Step 4: Return top final_num results
        final_results = reranked_results[:final_num]
        
        # Format results for saving
        formatted_results = []
        for r in final_results:
            formatted_results.append({
                'id': r['id'],
                'text': r['text'],
                'metadata': r['metadata'],
                'score': r['score'],
                'original_score': r.get('original_score'),
                'vector_score': r.get('vector_score'),
                'bm25_score': r.get('bm25_score'),
                'keyword_matches': r.get('keyword_matches', 0),
                'matched_keywords': r.get('matched_keywords', []),
                'match_ratio': r.get('match_ratio', 0.0),
                'is_html': r.get('is_html', False),
            })
        
        # Step 5: Save results to file if target directory is set
        if _rt_target_dir:
            import json
            from pathlib import Path
            import os
            
            # Create target directory if it doesn't exist
            target_path = Path(_rt_target_dir)
            target_path.mkdir(parents=True, exist_ok=True)
            
            # Create filename from query (sanitize for filesystem)
            # Replace invalid characters and limit length
            safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_', '.')).strip()
            safe_query = safe_query[:100]  # Limit length
            if not safe_query:
                safe_query = "query"
            filename = f"{safe_query}.json"
            filepath = target_path / filename
            
            # Save results as JSON
            result_data = {
                'query': query,
                'package': package,
                'results': formatted_results,
                'stats': {
                    'vector_results_count': len(vector_results),
                    'bm25_results_count': len(bm25_results),
                    'merged_results_count': len(merged_results),
                    'final_results_count': len(final_results),
                }
            }
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, indent=2, ensure_ascii=False)
                logger.info("Retrieval results saved to: %s", filepath)
                
                # Return minimal confirmation instead of full results
                return {
                    'status': 'success',
                    'message': f'Retrieval completed and saved to {filename}',
                    'query': query,
                    'package': package,
                    'file_path': str(filepath),
                    'results_count': len(final_results),
                    'stats': {
                        'vector_results_count': len(vector_results),
                        'bm25_results_count': len(bm25_results),
                        'merged_results_count': len(merged_results),
                        'final_results_count': len(final_results),
                    }
                }
            except Exception as save_error:
                logger.error("Failed to save retrieval results: %s", str(save_error))
                # Fall through to return full results if save fails
        
        # If no target directory set, return full results (backward compatibility)
        return {
            'query': query,
            'package': package,
            'results': formatted_results,
            'stats': {
                'vector_results_count': len(vector_results),
                'bm25_results_count': len(bm25_results),
                'merged_results_count': len(merged_results),
                'final_results_count': len(final_results),
            }
        }
        
    except Exception as e:
        logger.error("Error in retrieve_hybrid: %s", str(e))
        raise


@mcp.tool()
def retrieve_hybrid_multi(
    query: str,
    packages: List[str],
) -> Dict[str, Any]:
    """
    Perform hybrid retrieval (vector + BM25) with reranking across multiple packages.
    
    This tool performs hybrid retrieval for each specified package, then merges and reranks
    all results together. It combines vector similarity search and BM25 keyword matching for
    each package, merges results across packages, and applies keyword-based reranking.
    
    **Behavior:**
    - If the server was started with `--rt-target-dir`, retrieval results are automatically
      saved to JSON files in that directory. The tool returns a minimal confirmation message
      instead of the full results, significantly improving performance by avoiding the overhead
      of passing large result dictionaries back to the LLM.
    - If `--rt-target-dir` was not set, the tool returns the full results as a dictionary
      (default behavior for backward compatibility).
    
    Retrieval counts and weights are loaded from config.json and are not exposed as
    tool inputs.
    
    Args:
        query: The search query text
        packages: List of package names (e.g. ["itensormps", "itensors", "netket"]).
                  An empty list searches all available packages.
    
    Returns:
        **If `--rt-target-dir` is set (automatic file saving mode):**
        Dictionary containing:
        - status: "success"
        - message: Confirmation message with filename
        - query: The original query
        - packages: List of package names searched
        - file_path: Path to the saved JSON file
        - results_count: Number of results saved
        - stats: Statistics about the retrieval:
            - total_vector_results_count: Total number of vector results across all packages
            - total_bm25_results_count: Total number of BM25 results across all packages
            - merged_results_count: Number of merged results after combining packages
            - final_results_count: Number of final results after reranking
            - per_package_stats: Dictionary mapping package names to their individual stats
        
        **If `--rt-target-dir` is not set (default mode):**
        Dictionary containing:
        - query: The original query
        - packages: List of package names searched
        - results: List of retrieval results, each containing:
            - id: Chunk ID
            - text: Document text
            - metadata: Metadata (source_file, section_title, section_id, package)
            - score: Final reranked score (similarity form, higher is better, normalized to [0, 1])
            - original_score: Original retrieval score (similarity form, higher is better, normalized to [0, 1])
            - vector_score: Normalized vector similarity score (similarity form, higher is better, normalized to [0, 1], if available)
            - bm25_score: Normalized BM25 score (similarity form, higher is better, normalized to [0, 1], if available)
            
            **Note**: All scores are normalized to [0, 1] range in similarity form (higher is better).
            Results are sorted by score in descending order (highest similarity first).
            - keyword_matches: Number of matched keywords
            - matched_keywords: List of matched keywords
            - match_ratio: Ratio of matched keywords
            - is_html: Whether source file is HTML
        - stats: Statistics about the retrieval (same structure as above)
    """
    vector_num = None
    bm25_num = None
    final_num = None
    vector_weight = None
    reranker_weight = None
    html_boost = None

    try:
        # Coerce packages from string to list if needed (LLM may pass JSON string)
        if packages is not None and isinstance(packages, str):
            try:
                packages = json.loads(packages)
            except json.JSONDecodeError:
                raise ValueError(f"packages must be a list or valid JSON string, got: {packages!r}")
            if not isinstance(packages, list):
                raise ValueError(f"packages JSON must parse to a list, got: {type(packages).__name__}")

        # Load configuration from file (if not already loaded)
        config = _get_retrieval_config()
        allow_override = get_allow_parameter_override()
        
        # Log loaded configuration for debugging
        logger.debug("Loaded retrieval config: %s", config)
        
        # Packages parameter: If not provided, use all available packages
        configs = get_package_configs()
        if not packages:
            packages = list(configs.keys())
        
        # Validate packages
        invalid_packages = [pkg for pkg in packages if pkg not in configs]
        if invalid_packages:
            raise ValueError(
                f"Unknown package name(s): {invalid_packages}. "
                f"Available packages: {list(configs.keys())}"
            )
        
        # Numeric parameters: Use values from config.json (same logic as retrieve_hybrid)
        config_vector_num = config.get("vector_num")
        config_bm25_num = config.get("bm25_num")
        config_final_num = config.get("final_num")
        config_vector_weight = config.get("vector_weight")
        config_reranker_weight = config.get("reranker_weight")
        config_html_boost = config.get("html_boost")
        
        logger.debug("Config values from config.json: final_num=%s, vector_num=%s, bm25_num=%s", 
                     config_final_num, config_vector_num, config_bm25_num)
        
        if not allow_override:
            logger.info("Parameter override disabled in config.json. Using config file values for numeric parameters (ignoring any provided parameters).")
            vector_num = config_vector_num
            bm25_num = config_bm25_num
            final_num = config_final_num
            vector_weight = config_vector_weight
            reranker_weight = config_reranker_weight
            html_boost = config_html_boost
        else:
            vector_num = config_vector_num if vector_num is None else vector_num
            bm25_num = config_bm25_num if bm25_num is None else bm25_num
            final_num = config_final_num if final_num is None else final_num
            vector_weight = config_vector_weight if vector_weight is None else vector_weight
            reranker_weight = config_reranker_weight if reranker_weight is None else reranker_weight
            html_boost = config_html_boost if html_boost is None else html_boost
        
        logger.info("Performing multi-package hybrid retrieval for packages %s with query: %s", packages, query[:100])
        logger.debug("Using parameters: vector_num=%d, bm25_num=%d, final_num=%d, vector_weight=%.2f, reranker_weight=%.2f, html_boost=%.2f",
                     vector_num, bm25_num, final_num, vector_weight, reranker_weight, html_boost)
        
        # Step 1: Perform hybrid retrieval for each package
        all_package_merged_results = []
        per_package_stats = {}
        total_vector_count = 0
        total_bm25_count = 0
        
        for package in packages:
            logger.debug("Retrieving from package: %s", package)
            
            try:
                # Dual-channel retrieval for this package
                vector_results = retrieve_vector(query, k=vector_num, package=package, chroma_path=_chroma_path)
                bm25_results = retrieve_bm25(query, k=bm25_num, package=package, chroma_path=_chroma_path)
                
                total_vector_count += len(vector_results)
                total_bm25_count += len(bm25_results)
                
                # Merge results from both channels for this package
                merged_results = merge_retrieval_results(
                    vector_results,
                    bm25_results,
                    vector_weight=vector_weight
                )
                
                all_package_merged_results.append(merged_results)
                
                # Store per-package stats
                per_package_stats[package] = {
                    'vector_results_count': len(vector_results),
                    'bm25_results_count': len(bm25_results),
                    'merged_results_count': len(merged_results),
                }
            except Exception as e:
                # If retrieval fails for one package, log error and continue with other packages
                logger.error("Error retrieving from package '%s': %s", package, str(e))
                per_package_stats[package] = {
                    'vector_results_count': 0,
                    'bm25_results_count': 0,
                    'merged_results_count': 0,
                    'error': str(e)
                }
                # Add empty results for this package to avoid breaking the merge
                all_package_merged_results.append([])
        
        # Step 2: Merge results from all packages and re-normalize scores
        all_merged_results = merge_multi_package_results(
            all_package_merged_results,
            vector_weight=vector_weight
        )
        logger.debug("Merged results from all packages: %d unique chunks", len(all_merged_results))
        
        # Step 3: Keyword reranking with HTML priority (across all packages)
        reranked_results = keyword_reranker(
            all_merged_results,
            query,
            reranker_weight=reranker_weight,
            html_boost=html_boost
        )
        
        # Step 4: Return top final_num results
        final_results = reranked_results[:final_num]
        
        # Format results for saving
        formatted_results = []
        for r in final_results:
            formatted_results.append({
                'id': r['id'],
                'text': r['text'],
                'metadata': r['metadata'],
                'score': r['score'],
                'original_score': r.get('original_score'),
                'vector_score': r.get('vector_score'),
                'bm25_score': r.get('bm25_score'),
                'keyword_matches': r.get('keyword_matches', 0),
                'matched_keywords': r.get('matched_keywords', []),
                'match_ratio': r.get('match_ratio', 0.0),
                'is_html': r.get('is_html', False),
            })
        
        # Step 5: Save results to file if target directory is set
        if _rt_target_dir:
            import json
            from pathlib import Path
            
            # Create target directory if it doesn't exist
            target_path = Path(_rt_target_dir)
            target_path.mkdir(parents=True, exist_ok=True)
            
            # Create filename from query (sanitize for filesystem)
            safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_', '.')).strip()
            safe_query = safe_query[:100]  # Limit length
            if not safe_query:
                safe_query = "query"
            filename = f"{safe_query}.json"
            filepath = target_path / filename
            
            # Save results as JSON
            result_data = {
                'query': query,
                'packages': packages,
                'results': formatted_results,
                'stats': {
                    'total_vector_results_count': total_vector_count,
                    'total_bm25_results_count': total_bm25_count,
                    'merged_results_count': len(all_merged_results),
                    'final_results_count': len(final_results),
                    'per_package_stats': per_package_stats,
                }
            }
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, indent=2, ensure_ascii=False)
                logger.info("Multi-package retrieval results saved to: %s", filepath)
                
                # Return minimal confirmation instead of full results
                return {
                    'status': 'success',
                    'message': f'Multi-package retrieval completed and saved to {filename}',
                    'query': query,
                    'packages': packages,
                    'file_path': str(filepath),
                    'results_count': len(final_results),
                    'stats': {
                        'total_vector_results_count': total_vector_count,
                        'total_bm25_results_count': total_bm25_count,
                        'merged_results_count': len(all_merged_results),
                        'final_results_count': len(final_results),
                        'per_package_stats': per_package_stats,
                    }
                }
            except Exception as save_error:
                logger.error("Failed to save multi-package retrieval results: %s", str(save_error))
                # Fall through to return full results if save fails
        
        # If no target directory set, return full results (backward compatibility)
        return {
            'query': query,
            'packages': packages,
            'results': formatted_results,
            'stats': {
                'total_vector_results_count': total_vector_count,
                'total_bm25_results_count': total_bm25_count,
                'merged_results_count': len(all_merged_results),
                'final_results_count': len(final_results),
                'per_package_stats': per_package_stats,
            }
        }
        
    except Exception as e:
        logger.error("Error in retrieve_hybrid_multi: %s", str(e))
        raise


@mcp.tool()
def list_available_packages() -> Dict[str, Any]:
    """
    List all available packages for retrieval.
    
    Returns:
        Dictionary containing:
        - packages: List of available package names
        - default_package: Currently configured default package
        - package_info: Dictionary mapping package names to their collection names
    """
    try:
        configs = get_package_configs()
        package_info = {
            name: config["collection_name"]
            for name, config in configs.items()
        }
        
        return {
            'packages': list(configs.keys()),
            'default_package': _default_package,
            'package_info': package_info,
        }
    except Exception as e:
        logger.error("Error listing packages: %s", str(e))
        raise


def run_server(chroma_path: Optional[str] = None, default_package: Optional[str] = None, project_root: Optional[str] = None, rt_target_dir: Optional[str] = None) -> None:
    """Run the MCP server with the given configuration.
    
    Args:
        chroma_path: Path to ChromaDB storage directory
        default_package: Default package name to use
        project_root: Path to project root directory (auto-detected if not provided)
        rt_target_dir: Target directory for saving retrieval results (optional)
    """
    logger.info("Starting MCP Retrieve Server")
    
    # Set project root (auto-detect if not provided)
    if project_root:
        try:
            set_project_root(project_root)
            logger.info("Project root set to: %s", project_root)
        except Exception as e:
            logger.error("Failed to set project root: %s", str(e))
            raise
    else:
        try:
            detected_root = find_project_root()
            set_project_root(detected_root)
            logger.info("Project root auto-detected: %s", detected_root)
        except Exception as e:
            logger.error("Failed to auto-detect project root: %s", str(e))
            logger.error("Please provide --project-root argument")
            raise
    
    # Preload embedding model at startup (this may take a few seconds)
    # Note: Even if model is already downloaded, loading into memory takes time
    logger.info("Preloading embedding model (this may take a few seconds if first time, or ~2-5 seconds if cached)...")
    try:
        preload_embed_model()
        # Success message is already logged in preload_embed_model()
    except Exception as e:
        logger.error("Failed to preload embedding model: %s", str(e))
        logger.error("This may be due to network issues, missing dependencies, or insufficient memory.")
        raise
    
    # Load configuration from config.json
    logger.info("Loading configuration from config.json...")
    try:
        config = load_config()
        logger.info("Configuration loaded successfully")
        # Use config file default package if not provided via command line
        if not default_package:
            default_package = config.get("default_package", DEFAULT_PACKAGE)
    except Exception as e:
        logger.warning("Failed to load configuration file: %s. Using defaults.", str(e))
    
    # Set configuration
    if chroma_path:
        set_chroma_path(chroma_path)
    
    if default_package:
        set_default_package(default_package)
    
    if rt_target_dir:
        set_rt_target_dir(rt_target_dir)
    
    logger.info("MCP Retrieve Server ready")
    logger.info("Default package: %s", _default_package)
    if _chroma_path:
        logger.info("ChromaDB path: %s", _chroma_path)
    if _rt_target_dir:
        logger.info("Retrieval target directory: %s", _rt_target_dir)
    
    # Run the server
    mcp.run()
