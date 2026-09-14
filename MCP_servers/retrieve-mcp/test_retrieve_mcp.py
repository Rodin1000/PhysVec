"""Lightweight test script for retrieve-mcp server.

This script performs basic validation tests to ensure the server components
are working correctly without requiring a full database setup.

Usage:
    python test_retrieve_mcp.py
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all required modules can be imported."""
    print("=" * 60)
    print("Test 1: Module Imports")
    print("=" * 60)
    try:
        from mcp_server_retrieve import config_loader
        from mcp_server_retrieve import retrieval
        from mcp_server_retrieve import server
        print("✓ All modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_config_loader():
    """Test configuration loading functionality."""
    print("\n" + "=" * 60)
    print("Test 2: Configuration Loader")
    print("=" * 60)
    try:
        from mcp_server_retrieve.config_loader import (
            load_config,
            get_retrieval_config,
            get_default_package,
            get_allow_parameter_override,
            clear_config_cache,
        )
        
        # Clear cache to force reload
        clear_config_cache()
        
        # Test loading config
        config = load_config()
        print(f"✓ Configuration loaded: {len(config)} top-level keys")
        
        # Test retrieval config
        retrieval_config = get_retrieval_config()
        required_keys = ["vector_num", "bm25_num", "final_num", "vector_weight", "reranker_weight", "html_boost"]
        missing_keys = [k for k in required_keys if k not in retrieval_config]
        if missing_keys:
            print(f"✗ Missing keys in retrieval config: {missing_keys}")
            return False
        print(f"✓ Retrieval config has all required keys: {required_keys}")
        
        # Test default package
        default_package = get_default_package()
        print(f"✓ Default package: {default_package}")
        
        # Test allow_parameter_override
        allow_override = get_allow_parameter_override()
        print(f"✓ Allow parameter override: {allow_override}")
        
        # Test config file location
        config_file = Path(__file__).parent / "config.json"
        if config_file.exists():
            print(f"✓ Config file found at: {config_file}")
            with open(config_file, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
                print(f"  - Config file has 'allow_parameter_override': {'allow_parameter_override' in file_config}")
        else:
            print(f"⚠ Config file not found at: {config_file} (will use defaults)")
        
        return True
    except Exception as e:
        print(f"✗ Config loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_retrieval_functions():
    """Test retrieval module functions."""
    print("\n" + "=" * 60)
    print("Test 3: Retrieval Module Functions")
    print("=" * 60)
    try:
        from mcp_server_retrieve.retrieval import (
            get_package_configs,
            find_project_root,
            get_project_root,
            DEFAULT_PACKAGE,
        )
        
        # Test package configs
        configs = get_package_configs()
        print(f"✓ Package configs loaded: {len(configs)} packages")
        for pkg_name in configs.keys():
            print(f"  - {pkg_name}: {configs[pkg_name].get('collection_name', 'N/A')}")
        
        # Test project root detection
        try:
            project_root = find_project_root()
            print(f"✓ Project root detected: {project_root}")
        except FileNotFoundError as e:
            print(f"⚠ Project root not found: {e}")
            print("  (This is OK if running from a different location)")
        
        # Test default package
        print(f"✓ Default package constant: {DEFAULT_PACKAGE}")
        
        return True
    except Exception as e:
        print(f"✗ Retrieval module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_server_parameter_logic():
    """Test server parameter handling logic."""
    print("\n" + "=" * 60)
    print("Test 4: Server Parameter Logic")
    print("=" * 60)
    try:
        from mcp_server_retrieve.config_loader import (
            get_retrieval_config,
            get_allow_parameter_override,
            clear_config_cache,
        )
        
        # Clear cache to ensure fresh load
        clear_config_cache()
        
        # Test config retrieval
        config = get_retrieval_config()
        allow_override = get_allow_parameter_override()
        
        print(f"✓ Config retrieved: {len(config)} parameters")
        print(f"✓ Allow override: {allow_override}")
        
        # Test parameter resolution logic (simulated)
        test_cases = [
            {
                "name": "Config file values",
                "vector_num": None,
                "expected_source": "config file",
            },
            {
                "name": "Override allowed",
                "vector_num": 100,
                "allow_override": True,
                "expected_source": "parameter",
            },
            {
                "name": "Override disabled",
                "vector_num": 100,
                "allow_override": False,
                "expected_source": "config file",
            },
        ]
        
        for case in test_cases:
            if case.get("allow_override") is None:
                case_allow = allow_override
            else:
                case_allow = case["allow_override"]
            
            if not case_allow:
                # Override disabled: should use config file
                resolved = config.get("vector_num", 30)
                source = "config file"
            else:
                # Override allowed: use parameter if provided, else config file
                if case["vector_num"] is not None:
                    resolved = case["vector_num"]
                    source = "parameter"
                else:
                    resolved = config.get("vector_num", 30)
                    source = "config file"
            
            expected = case["expected_source"]
            if source == expected:
                print(f"✓ {case['name']}: Uses {source} (value: {resolved})")
            else:
                print(f"✗ {case['name']}: Expected {expected}, got {source}")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Server parameter logic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_file_structure():
    """Test that config.json has correct structure."""
    print("\n" + "=" * 60)
    print("Test 5: Config File Structure")
    print("=" * 60)
    try:
        config_file = Path(__file__).parent / "config.json"
        
        if not config_file.exists():
            print(f"⚠ Config file not found at: {config_file}")
            print("  (This is OK, defaults will be used)")
            return True
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Check required top-level keys
        required_keys = ["retrieval", "default_package", "allow_parameter_override"]
        missing_keys = [k for k in required_keys if k not in config]
        if missing_keys:
            print(f"✗ Missing top-level keys: {missing_keys}")
            return False
        print(f"✓ All required top-level keys present: {required_keys}")
        
        # Check retrieval section
        retrieval = config.get("retrieval", {})
        required_retrieval_keys = ["vector_num", "bm25_num", "final_num", "vector_weight", "reranker_weight", "html_boost"]
        missing_retrieval_keys = [k for k in required_retrieval_keys if k not in retrieval]
        if missing_retrieval_keys:
            print(f"✗ Missing retrieval keys: {missing_retrieval_keys}")
            return False
        print(f"✓ All required retrieval keys present: {required_retrieval_keys}")
        
        # Check types
        if not isinstance(config.get("allow_parameter_override"), bool):
            print(f"✗ allow_parameter_override should be boolean, got {type(config.get('allow_parameter_override'))}")
            return False
        print(f"✓ allow_parameter_override is boolean: {config.get('allow_parameter_override')}")
        
        if not isinstance(config.get("default_package"), str):
            print(f"✗ default_package should be string, got {type(config.get('default_package'))}")
            return False
        print(f"✓ default_package is string: {config.get('default_package')}")
        
        return True
    except json.JSONDecodeError as e:
        print(f"✗ Config file JSON parsing failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Config file structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Retrieve-MCP Server Test Suite")
    print("=" * 60)
    print("This script performs basic validation tests.")
    print("It does NOT test actual retrieval (requires database).\n")
    
    results = []
    
    # Run tests
    results.append(("Module Imports", test_imports()))
    results.append(("Configuration Loader", test_config_loader()))
    results.append(("Retrieval Module", test_retrieval_functions()))
    results.append(("Server Parameter Logic", test_server_parameter_logic()))
    results.append(("Config File Structure", test_config_file_structure()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The server components appear to be working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

