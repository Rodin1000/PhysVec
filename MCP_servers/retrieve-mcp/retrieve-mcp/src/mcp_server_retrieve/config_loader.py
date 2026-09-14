"""Configuration loader for MCP Retrieve Server.

This module loads configuration from config.json file located in the retrieve-mcp root directory.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_CONFIG = {
    "retrieval": {
        "vector_num": 30,
        "bm25_num": 30,
        "final_num": 3,
        "vector_weight": 0.7,
        "reranker_weight": 0.3,
        "html_boost": 0.15,
    },
    "default_package": "itensormps",
    "allow_parameter_override": True,
}

# Global cache for loaded configuration
_config_cache: Optional[Dict[str, Any]] = None


def find_retrieve_mcp_root() -> Optional[Path]:
    """
    Find the retrieve-mcp root directory by looking for config.json or src/mcp_server_retrieve.
    
    Returns:
        Path to retrieve-mcp root directory, or None if not found
    """
    # Try to find from current file location
    current_file = Path(__file__).resolve()
    # current_file is at: retrieve-mcp/src/mcp_server_retrieve/config_loader.py
    # So retrieve-mcp root is: current_file.parent.parent.parent
    candidate = current_file.parent.parent.parent
    
    # Check if config.json exists there
    config_file = candidate / "config.json"
    if config_file.exists():
        return candidate
    
    # Try searching from current working directory
    current = Path.cwd().resolve()
    for _ in range(10):  # Search up to 10 levels
        config_file = current / "config.json"
        src_dir = current / "src" / "mcp_server_retrieve"
        if config_file.exists() and src_dir.exists():
            return current
        if current == current.parent:
            break
        current = current.parent
    
    return None


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from config.json file.
    
    Args:
        config_path: Optional path to config.json file.
                    If None, searches for config.json in retrieve-mcp root directory.
    
    Returns:
        Configuration dictionary with defaults applied for missing values
    """
    global _config_cache
    
    # Return cached config if available
    if _config_cache is not None:
        return _config_cache
    
    # Find config file
    if config_path is None:
        retrieve_mcp_root = find_retrieve_mcp_root()
        if retrieve_mcp_root is None:
            logger.warning(
                "Could not find retrieve-mcp root directory. Using default configuration."
            )
            _config_cache = DEFAULT_CONFIG.copy()
            return _config_cache
        config_path = retrieve_mcp_root / "config.json"
    else:
        config_path = Path(config_path)
    
    # Load config file
    if not config_path.exists():
        logger.warning(
            f"Configuration file not found at {config_path}. Using default configuration."
        )
        _config_cache = DEFAULT_CONFIG.copy()
        return _config_cache
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            loaded_config = json.load(f)
        
        # Merge with defaults (defaults take precedence for missing keys)
        config = DEFAULT_CONFIG.copy()
        if "retrieval" in loaded_config:
            config["retrieval"].update(loaded_config["retrieval"])
        if "default_package" in loaded_config:
            config["default_package"] = loaded_config["default_package"]
        if "allow_parameter_override" in loaded_config:
            config["allow_parameter_override"] = loaded_config["allow_parameter_override"]
        
        _config_cache = config
        logger.info(f"Configuration loaded from {config_path}")
        return config
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse configuration file {config_path}: {e}")
        logger.warning("Using default configuration.")
        _config_cache = DEFAULT_CONFIG.copy()
        return _config_cache
    except Exception as e:
        logger.error(f"Failed to load configuration file {config_path}: {e}")
        logger.warning("Using default configuration.")
        _config_cache = DEFAULT_CONFIG.copy()
        return _config_cache


def get_retrieval_config() -> Dict[str, Any]:
    """
    Get retrieval configuration parameters.
    
    Returns:
        Dictionary containing retrieval parameters:
        - vector_num: Number of vector search results
        - bm25_num: Number of BM25 search results
        - final_num: Number of final results after reranking
        - vector_weight: Weight for vector channel in fusion
        - reranker_weight: Weight for keyword reranker
        - html_boost: Boost for HTML files
    """
    config = load_config()
    return config.get("retrieval", DEFAULT_CONFIG["retrieval"]).copy()


def get_default_package() -> str:
    """
    Get default package name from configuration.
    
    Returns:
        Default package name
    """
    config = load_config()
    return config.get("default_package", DEFAULT_CONFIG["default_package"])


def get_allow_parameter_override() -> bool:
    """
    Get allow_parameter_override setting from configuration.
    
    Returns:
        True if parameters can override config file values, False otherwise
    """
    config = load_config()
    return config.get("allow_parameter_override", DEFAULT_CONFIG["allow_parameter_override"])


def clear_config_cache() -> None:
    """Clear the configuration cache (useful for testing or reloading config)."""
    global _config_cache
    _config_cache = None

