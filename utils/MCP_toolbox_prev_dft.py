# This program utilize mcp-use package
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import os
import sys
import asyncio
import json
import logging
from dotenv import load_dotenv
from typing import Optional, Dict, List
from pydantic import Field, SecretStr
from pathlib import Path

# mcp-use package
from mcp_use import MCPAgent, MCPClient
from langchain_core.utils.utils import secret_from_env

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# config file in this project
from config import OPENROUTER_API_KEY, YIDONG_API_KEY
from utils.send_chat_yidong import MODEL_MAX_TOKENS

# Setup logging for MCP operations
logger = logging.getLogger(__name__)

class ChatOpenRouter(ChatOpenAI):
    openai_api_key: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=secret_from_env("OPENROUTER_API_KEY", default=None),
    )
    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"openai_api_key": "OPENROUTER_API_KEY"}

    def __init__(self,
                openai_api_key: Optional[str] = None,
                **kwargs):
        openai_api_key = (
            openai_api_key or os.environ.get("OPENROUTER_API_KEY")
        )
        super().__init__(
            base_url="https://openrouter.ai/api/v1",
            openai_api_key=openai_api_key,
            **kwargs
        )


class ChatYidongAnthropic(ChatAnthropic):
    """Chat model for yidong API using Anthropic Messages API format."""
    
    def __init__(self,
                model: str = None,
                anthropic_api_key: Optional[str] = None,
                temperature: float = 0.2,
                max_tokens: int = None,
                **kwargs):
        # yidong API uses Anthropic Messages API format at /v1/messages endpoint
        # ChatAnthropic uses anthropic_api_url parameter for custom endpoints
        # ChatAnthropic automatically adds "Bearer" prefix, so remove it if already present
        
        # prepare max_tokens
        if max_tokens is None:
            if model not in MODEL_MAX_TOKENS:
                raise ValueError(f"Model {model} not found in MODEL_MAX_TOKENS")
            max_tokens = MODEL_MAX_TOKENS[model]

        api_key = anthropic_api_key or os.environ.get("YIDONG_API_KEY") or YIDONG_API_KEY
        if api_key and api_key.startswith("Bearer "):
            api_key = api_key[7:]  # Remove "Bearer " prefix
        super().__init__(
            model=model,
            anthropic_api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            anthropic_api_url="https://yinli.one",
            **kwargs
        )


class ChatYidongOpenAI(ChatOpenAI):
    """Chat model for yidong API using OpenAI-compatible API format."""
    
    def __init__(self,
                model: str = None,
                openai_api_key: Optional[str] = None,
                temperature: float = 0.2,
                max_tokens: int = None,
                **kwargs):
        # yidong API uses OpenAI-compatible API format at /v1/chat/completions endpoint
        # ChatOpenAI uses base_url parameter for custom endpoints
        # ChatOpenAI automatically adds "/chat/completions" to base_url
        # ChatOpenAI automatically adds "Bearer" prefix to API key, so remove it if already present
        
        # prepare max_tokens
        if max_tokens is None:
            if model not in MODEL_MAX_TOKENS:
                raise ValueError(f"Model {model} not found in MODEL_MAX_TOKENS")
            max_tokens = MODEL_MAX_TOKENS[model]

        api_key = openai_api_key or os.environ.get("YIDONG_API_KEY") or YIDONG_API_KEY
        if api_key and api_key.startswith("Bearer "):
            api_key = api_key[7:]  # Remove "Bearer " prefix
        super().__init__(
            model=model,
            openai_api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url="https://yinli.one/v1",  # ChatOpenAI will append /chat/completions
            **kwargs
        )


async def send_chat_through_mcp_use(user_model: str = "deepseek/deepseek-chat-v3.1:free", model_temp: float = 0.1, user_prompt: str = "User input missing!", max_steps: int = 25) -> str:
    # Load environment variables
    load_dotenv()

    # Create configuration dictionary
    # config = {
    #   "mcpServers": {
    #     "playwright": {
    #       "command": "npx",
    #       "args": ["@playwright/mcp@latest", "--browser=firefox"],
    #       "env": {
    #         "DISPLAY": ":1"
    #       }
    #     }
    #   }
    # }
    # client = MCPClient.from_dict(config)
    client = MCPClient.from_config_file("../MCP_servers/MCP_server_config.json")

    # Creat a LLM object
    llm = ChatOpenRouter(
        model_name=user_model,
        temperature=model_temp,
    )

    # Create agent with the client & LLM object
    agent = MCPAgent(llm=llm, client=client, max_steps=max_steps)

    # Code generation prompt
    response = await agent.run(user_prompt,) 
    return response


async def send_chat_through_mcp_dynamic(
    user_model: str = "deepseek/deepseek-chat-v3.1:free",
    model_temp: float = 0.1,
    user_prompt: str = "User input missing!",
    max_steps: int = 50,
    # API backend selection
    api_backend: Optional[str] = None,                      # "openrouter" or "yidong". If None, auto-detect from user_model
    # MCP server selection
    init_servers: Optional[List[str]] = None,                # List of servers to initialize, e.g. ["filesystem-mcp", "retrieve-mcp"]
                                                             # Default: ["filesystem-mcp"]
    # Dynamic filesystem params (only used if "filesystem-mcp" in init_servers)
    fs_target_dir: Optional[str] = None,                     # absolute or relative path; becomes --project-dir
    split_repo: Optional[Dict[str, str]] = None,             # {"pkg/mod.py": "..."} if you want LLM to write files
    references: Optional[Dict[str, str]] = None,             # {"libs": "../Human_code_library"}
    base_mcp_config_file: Optional[str] = None,             # Base config file to merge
    override_filesystem: bool = True,                        # override existing 'filesystem' entry in base config
    # Retrieve-mcp parameters (only used if "retrieve-mcp" in init_servers)
    retrieve_chroma_path: Optional[str] = None,              # ChromaDB path (default: "authority_library/rag_store/rag_chroma")
    retrieve_package: Optional[str] = None,                  # default package (default: "itensormps")
    retrieve_project_root: Optional[str] = None,             # project root (auto-detected if None)
    retrieve_rt_target_dir: Optional[str] = None,            # target directory for saving retrieval results (if None, results returned to LLM)
    retrieve_log_level: Optional[str] = None,                # log level (uses filesystem log_level if None)
    override_retrieve: bool = True,                          # override existing 'retrieve' entry in base config
    # Program-mcp parameters (only used if "program-mcp" in init_servers)
    program_path: Optional[str] = None,                     # path to the program file
    program_timeout: int = 30,                              # default timeout for code execution (seconds)
    program_log_level: Optional[str] = None,                # log level (uses filesystem log_level if None)
    override_program: bool = True,                           # override existing 'program' entry in base config
    # Program-dft-mcp parameters (only used if "program-dft-mcp" in init_servers)
    program_dft_input_path: Optional[str] = None,            # path to ORCA input file (.inp)
    program_dft_timeout: int = 600,                          # default timeout for ORCA execution (seconds)
    override_program_dft: bool = True,                      # override existing 'program-dft' entry in base config
) -> str:
    """
    Send chat through MCP with dynamically configured servers.
    
    This function allows you to dynamically configure which MCP servers to use.
    You can specify which servers to initialize via the init_servers parameter.
    
    Args:
        api_backend: API backend to use. Options:
                    - "openrouter": Use OpenRouter API (default for most models)
                    - "yidong": Use Yidong API (yinli.one)
                    - None: Auto-detect from user_model (if model is in MODEL_MAX_TOKENS, use "yidong", else "openrouter")
        user_model: Model identifier. For yidong models, use model names from MODEL_MAX_TOKENS.
        init_servers: List of server names to initialize. Supported values:
                     - "filesystem-mcp": File system operations
                     - "retrieve-mcp": RAG retrieval operations
                     - "program-mcp": Code execution (Python/Julia)
                     - "program-dft-mcp": ORCA/DFT program execution
                     Default: ["filesystem-mcp"]
        
        Other parameters are only used if the corresponding server is in init_servers.
    """
    load_dotenv()

    # Set default init_servers
    if init_servers is None:
        init_servers = ["filesystem-mcp"]
    
    # Validate init_servers
    valid_servers = {"filesystem-mcp", "retrieve-mcp", "program-mcp", "program-dft-mcp"}
    invalid_servers = set(init_servers) - valid_servers
    if invalid_servers:
        raise ValueError(f"Invalid server names in init_servers: {invalid_servers}. Valid options: {valid_servers}")
    
    # Check if filesystem-mcp is requested but fs_target_dir is not provided
    if "filesystem-mcp" in init_servers and not fs_target_dir:
        raise ValueError("fs_target_dir must be provided when 'filesystem-mcp' is in init_servers.")
    
    # Check if program-mcp is requested but program_path is not provided
    if "program-mcp" in init_servers and not program_path:
        raise ValueError("program_path must be provided when 'program-mcp' is in init_servers.")
    
    # Check if program-dft-mcp is requested but program_dft_input_path is not provided
    if "program-dft-mcp" in init_servers and not program_dft_input_path:
        raise ValueError("program_dft_input_path must be provided when 'program-dft-mcp' is in init_servers.")
    
    # If only retrieve-mcp is requested, fs_target_dir is optional (can be None)

    # 1) Build MCP client (dynamic) — LLM will still call tools directly
    if "filesystem-mcp" in init_servers:
        # Pre-create parent directories for all files (safer for this server)
        if split_repo:
            ensure_parents_exist(fs_target_dir, list(split_repo.keys()))

    logger.info("Building dynamic MCP configuration...")
    logger.info(f"Initializing servers: {init_servers}")
    
    dynamic_config = build_dynamic_config(
        target_dir=fs_target_dir if "filesystem-mcp" in init_servers else None,
        base_config_file=base_mcp_config_file,
        references=references,
        log_level="INFO",
        console_only=False,  # Don't pollute stdout with logs (breaks JSONRPC)
        server_name="filesystem",
        override_filesystem=override_filesystem,
        # Server selection
        init_servers=init_servers,
        # Retrieve-mcp parameters
        retrieve_chroma_path=retrieve_chroma_path,
        retrieve_package=retrieve_package,
        retrieve_project_root=retrieve_project_root,
        retrieve_rt_target_dir=retrieve_rt_target_dir,
        retrieve_log_level=retrieve_log_level,
        override_retrieve=override_retrieve,
        # Program-mcp parameters
        program_path=program_path,
        program_timeout=program_timeout,
        program_log_level=program_log_level,
        override_program=override_program,
        # Program-dft-mcp parameters
        program_dft_input_path=program_dft_input_path,
        program_dft_timeout=program_dft_timeout,
        override_program_dft=override_program_dft,
    )
    
    logger.info(f"Initializing MCP client with {len(dynamic_config.get('mcpServers', {}))} server(s)...")
    logger.debug(f"MCP servers: {list(dynamic_config.get('mcpServers', {}).keys())}")
    
    # Debug: Log the actual configuration (with env vars masked for security)
    config_for_log = json.dumps(dynamic_config, indent=2, default=str)
    logger.debug(f"MCP configuration:\n{config_for_log}")
    
    try:
        client = MCPClient.from_dict(dynamic_config)
        logger.info("MCP client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize MCP client: {e}")
        logger.error("This may be due to:")
        logger.error("  1. MCP server commands not found in PATH (mcp-server-filesystem, mcp-server-retrieve)")
        logger.error("  2. Server startup timeout or failure")
        logger.error("  3. Missing dependencies or configuration issues")
        raise

    # 2) LLM - Choose based on api_backend or auto-detect from user_model
    # Auto-detect api_backend if not specified
    if api_backend is None:
        # If model is in MODEL_MAX_TOKENS (yidong models), use yidong backend
        if user_model in MODEL_MAX_TOKENS:
            api_backend = "yidong"
        else:
            api_backend = "openrouter"
    
    # Route to appropriate LLM based on api_backend
    if api_backend == "yidong":
        # Models that use OpenAI-compatible API format via yidong
        openai_format_models = ["grok-4-fast-reasoning"]
        if user_model in openai_format_models:
            # grok models use OpenAI-compatible API format via yidong
            llm = ChatYidongOpenAI(
                model=user_model,
                temperature=model_temp,
                openai_api_key=YIDONG_API_KEY
            )
        else:
            # Other models (claude, gemini, etc.) use Anthropic Messages API format via yidong
            llm = ChatYidongAnthropic(
                model=user_model,
                temperature=model_temp,
                anthropic_api_key=YIDONG_API_KEY
            )
    else:  # api_backend == "openrouter" (default)
        llm = ChatOpenRouter(model_name=user_model, temperature=model_temp)

    # 3) Agent with MCP client
    agent = MCPAgent(llm=llm, client=client, max_steps=max_steps)

    try:
        response = await agent.run(user_prompt,)
        return response
    except Exception as e:
        import logging
        logging.error(f"Error in MCP agent run: {type(e)} {repr(e)}")
        return {"error": "MCP agent exception", "detail": str(e)}


# Write a dynamic MCP config for filesystem server
def _fs_server_entry(
    target_dir: str,
    references: Optional[Dict[str, str]] = None,
    log_level: str = "INFO",
    console_only: bool = False,  # Changed default to False to avoid stdout pollution
    log_file: Optional[str] = None,
) -> dict:
    root = str(Path(target_dir).expanduser().resolve())
    args = ["--project-dir", root]
    if references is not None:
        for name, p in references.items():
            args += ["--reference-project", f"{name}={str(Path(p).expanduser().resolve())}"]
    if log_level:
        args += ["--log-level", log_level]
    
    # IMPORTANT: Don't use --console-only as it outputs logs to stdout,
    # which pollutes the JSONRPC communication channel (MCP uses stdio)
    if console_only:
        args += ["--console-only"]
    elif log_file:
        # Use custom log file path
        args += ["--log-file", str(Path(log_file).expanduser().resolve())]
    # If neither console_only nor log_file is specified, use default behavior
    # (logs to project_dir/logs/mcp_filesystem_server_{timestamp}.log)
    
    # Set UTF-8 encoding for Windows to prevent UnicodeEncodeError
    # when mcp-server-filesystem prints Unicode characters (✓, ⚠, etc.)
    import platform
    result = {
        "command": "mcp-server-filesystem",
        "args": args,
    }
    if platform.system() == "Windows":
        # Merge with parent environment to ensure other env vars are preserved
        import os
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        result["env"] = env
    
    return result


# Write MCP config for retrieve server
def _retrieve_server_entry(
    chroma_path: Optional[str] = None,
    package: Optional[str] = None,
    project_root: Optional[str] = None,
    rt_target_dir: Optional[str] = None,
    log_level: str = "INFO",
) -> dict:
    """
    Create retrieve-mcp server entry.
    
    Args:
        chroma_path: ChromaDB storage path (relative to project root, default: "authority_library/rag_store/rag_chroma")
        package: Default package name (default: "itensormps")
        project_root: Project root directory (optional, auto-detected if not provided)
        rt_target_dir: Target directory for saving retrieval results (optional). If set, results are saved to files instead of returned to LLM.
        log_level: Logging level (default: "INFO")
    """
    args = []
    if chroma_path:
        args += ["--chroma-path", chroma_path]
    if package:
        args += ["--package", package]
    if project_root:
        args += ["--project-root", str(Path(project_root).expanduser().resolve())]
    if rt_target_dir:
        args += ["--rt-target-dir", str(Path(rt_target_dir).expanduser().resolve())]
    if log_level:
        args += ["--log-level", log_level]
    
    # Set UTF-8 encoding for Windows to prevent UnicodeEncodeError
    import platform
    result = {
        "command": "mcp-server-retrieve",
        "args": args,
    }
    if platform.system() == "Windows":
        # Merge with parent environment to ensure other env vars are preserved
        import os
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        result["env"] = env
    
    return result


# Write MCP config for program server
def _program_server_entry(
    program_path: str,
    timeout: int = 30,
    log_level: str = "INFO",
) -> dict:
    """
    Create program-mcp server entry.
    
    Args:
        program_path: Path to the program file (required)
        timeout: Default timeout in seconds (default: 30)
        log_level: Logging level (default: "INFO")
    """
    args = [
        "-m", "mcp_server_program.main",
        "--program-path", str(Path(program_path).expanduser().resolve()),
        "--timeout", str(timeout),
        "--log-level", log_level,
    ]
    
    # Get the path to program-mcp source directory
    program_mcp_dir = Path(__file__).parent.parent / "MCP_servers" / "program-mcp"
    
    # Set UTF-8 encoding for Windows to prevent UnicodeEncodeError
    import platform
    import os
    result = {
        "command": "python",
        "args": args,
        "cwd": str(program_mcp_dir.resolve()),
    }
    if platform.system() == "Windows":
        # Merge with parent environment to ensure other env vars are preserved
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        # Add src directory to PYTHONPATH so Python can find mcp_server_program module
        pythonpath = env.get("PYTHONPATH", "")
        src_dir = str(program_mcp_dir / "src")
        if pythonpath:
            env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{pythonpath}"
        else:
            env["PYTHONPATH"] = src_dir
        result["env"] = env
    else:
        # For Unix/MacOS, also set PYTHONPATH
        env = dict(os.environ)
        src_dir = str(program_mcp_dir / "src")
        pythonpath = env.get("PYTHONPATH", "")
        if pythonpath:
            env["PYTHONPATH"] = f"{src_dir}:{pythonpath}"
        else:
            env["PYTHONPATH"] = src_dir
        result["env"] = env
    
    return result


# Write MCP config for program-dft server
def _program_dft_server_entry(
    input_path: str,
    timeout: int = 600,
) -> dict:
    """
    Create program-dft-mcp server entry.
    
    Args:
        input_path: Path to the ORCA input file (.inp) (required)
        timeout: Default timeout in seconds (default: 600)
    """
    args = [
        "-m", "mcp_server_program_dft.main",
        "--input-path", str(Path(input_path).expanduser().resolve()),
        "--timeout", str(timeout),
    ]
    
    # Get the path to program-dft-mcp source directory
    program_dft_mcp_dir = Path(__file__).parent.parent / "MCP_servers" / "program-dft-mcp"
    
    # Set UTF-8 encoding for Windows to prevent UnicodeEncodeError
    import platform
    import os
    result = {
        "command": "python",
        "args": args,
        "cwd": str(program_dft_mcp_dir.resolve()),
    }
    if platform.system() == "Windows":
        # Merge with parent environment to ensure other env vars are preserved
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        # Add src directory to PYTHONPATH so Python can find mcp_server_program_dft module
        pythonpath = env.get("PYTHONPATH", "")
        src_dir = str(program_dft_mcp_dir / "src")
        if pythonpath:
            env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{pythonpath}"
        else:
            env["PYTHONPATH"] = src_dir
        result["env"] = env
    else:
        # For Unix/MacOS, also set PYTHONPATH
        env = dict(os.environ)
        src_dir = str(program_dft_mcp_dir / "src")
        pythonpath = env.get("PYTHONPATH", "")
        if pythonpath:
            env["PYTHONPATH"] = f"{src_dir}:{pythonpath}"
        else:
            env["PYTHONPATH"] = src_dir
        result["env"] = env
    
    return result


def find_project_root_from_utils() -> Optional[Path]:
    """
    Find project root by looking for authority_library directory.
    This is a helper function to ensure retrieve-mcp can find the project root.
    """
    # Start from utils directory (where this file is located)
    current = Path(__file__).parent.parent.resolve()
    
    # Search upward for authority_library
    while current != current.parent:
        authority_lib = current / "authority_library"
        if authority_lib.exists() and authority_lib.is_dir():
            return current
        current = current.parent
    
    return None


def build_dynamic_config(
    target_dir: Optional[str] = None,
    base_config_file: Optional[str] = None,
    references: Optional[Dict[str, str]] = None,
    log_level: str = "INFO",
    console_only: bool = False,  # Default changed to False to avoid stdout pollution
    server_name: str = "filesystem",
    override_filesystem: bool = True,
    # Server selection
    init_servers: Optional[List[str]] = None,                # List of servers to initialize
    # Retrieve-mcp parameters
    retrieve_chroma_path: Optional[str] = None,
    retrieve_package: Optional[str] = None,
    retrieve_project_root: Optional[str] = None,
    retrieve_rt_target_dir: Optional[str] = None,
    retrieve_log_level: Optional[str] = None,
    override_retrieve: bool = True,
    # Program-mcp parameters
    program_path: Optional[str] = None,
    program_timeout: int = 30,
    program_log_level: Optional[str] = None,
    override_program: bool = True,
    # Program-dft-mcp parameters
    program_dft_input_path: Optional[str] = None,
    program_dft_timeout: int = 600,
    override_program_dft: bool = True,
) -> dict:
    """
    Return an 'mcpServers' config dict that mcp-use understands.
    Builds configuration based on init_servers list to determine which servers to include.
    
    Args:
        target_dir: Target directory for filesystem server (required if "filesystem-mcp" in init_servers)
        base_config_file: Base MCP config file to merge (optional)
        references: Reference projects for filesystem server
        log_level: Log level for servers
        console_only: Console only mode for filesystem server (Default: False to avoid JSONRPC pollution)
                     WARNING: Setting to True will output logs to stdout, which can break MCP communication
        server_name: Name for filesystem server entry
        override_filesystem: Whether to override existing filesystem entry
        init_servers: List of server names to initialize (e.g., ["filesystem-mcp", "retrieve-mcp"])
                     Default: ["filesystem-mcp"]
        retrieve_chroma_path: ChromaDB path for retrieve server (default: "authority_library/rag_store/rag_chroma")
        retrieve_package: Default package for retrieve server (default: "itensormps")
        retrieve_project_root: Project root for retrieve server (auto-detected if None)
        retrieve_rt_target_dir: Target directory for saving retrieval results (if None, results returned to LLM)
        retrieve_log_level: Log level for retrieve server (uses log_level if None)
        override_retrieve: Whether to override existing retrieve entry (default: True)
        program_path: Path to the program file for program-mcp (required if "program-mcp" in init_servers)
        program_timeout: Default timeout for code execution (default: 30)
        program_log_level: Log level for program-mcp (uses log_level if None)
        override_program: Whether to override existing program entry (default: True)
        program_dft_input_path: Path to ORCA input file for program-dft-mcp (required if "program-dft-mcp" in init_servers)
        program_dft_timeout: Default timeout for ORCA execution (default: 600)
        override_program_dft: Whether to override existing program-dft entry (default: True)
    """
    # Set default init_servers
    if init_servers is None:
        init_servers = ["filesystem-mcp"]
    
    config: dict = {"mcpServers": {}}
    
    # Resolve base_config_file to absolute path if provided
    if base_config_file:
        base_config_path = Path(base_config_file)
        if not base_config_path.is_absolute():
            # Resolve relative to PROJECT_ROOT
            base_config_path = Path(PROJECT_ROOT) / base_config_path
        base_config_path = base_config_path.resolve()
        
        if base_config_path.exists():
            try:
                with open(base_config_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict) and "mcpServers" in loaded:
                        config["mcpServers"] = dict(loaded["mcpServers"])  # shallow copy
                        logger.debug(f"Loaded base config from {base_config_path}")
            except Exception as e:
                logger.warning(f"Failed to load base config file {base_config_path}: {e}")
        else:
            logger.warning(f"Base config file not found: {base_config_path}, using defaults")

    # Add filesystem server entry if requested
    if "filesystem-mcp" in init_servers:
        if target_dir is None:
            raise ValueError("target_dir must be provided when 'filesystem-mcp' is in init_servers")
        
        fs_entry = _fs_server_entry(
            target_dir=target_dir,
            references=references,
            log_level=log_level,
            console_only=False,  # Don't pollute stdout with logs (breaks JSONRPC)
            log_file=None,  # Use default log file location
        )

        if override_filesystem or server_name not in config["mcpServers"]:
            config["mcpServers"][server_name] = fs_entry
        else:
            # If not overriding, add a uniquely named server (e.g., "filesystem_job123")
            # but usually you want to override to avoid tool ambiguity
            raise ValueError(f"Server name '{server_name}' already exists; set override_filesystem=True or use a new name.")
    
    # Add retrieve-mcp server entry if requested
    if "retrieve-mcp" in init_servers:
        # Auto-detect project root if not provided (for retrieve-mcp)
        if retrieve_project_root is None:
            detected_root = find_project_root_from_utils()
            if detected_root:
                retrieve_project_root = str(detected_root)
                logger.debug(f"Auto-detected project root for retrieve-mcp: {retrieve_project_root}")
            else:
                logger.warning("Could not auto-detect project root for retrieve-mcp. Server will try to auto-detect at startup.")
        
        retrieve_entry = _retrieve_server_entry(
            chroma_path=retrieve_chroma_path,
            package=retrieve_package,
            project_root=retrieve_project_root,
            rt_target_dir=retrieve_rt_target_dir,
            log_level=retrieve_log_level or log_level,
        )
        
        retrieve_server_name = "retrieve"
        if override_retrieve or retrieve_server_name not in config["mcpServers"]:
            config["mcpServers"][retrieve_server_name] = retrieve_entry
            logger.debug(f"Added retrieve-mcp server with project_root: {retrieve_project_root}")
        else:
            # If retrieve server already exists and override is False, just log a warning
            # (unlike filesystem, we don't raise an error to allow flexibility)
            logger.warning(f"Retrieve server '{retrieve_server_name}' already exists; not overriding. Set override_retrieve=True to override.")
    
    # Add program-mcp server entry if requested
    if "program-mcp" in init_servers:
        if program_path is None:
            raise ValueError("program_path must be provided when 'program-mcp' is in init_servers")
        
        program_entry = _program_server_entry(
            program_path=program_path,
            timeout=program_timeout,
            log_level=program_log_level or log_level,
        )
        
        program_server_name = "program"
        if override_program or program_server_name not in config["mcpServers"]:
            config["mcpServers"][program_server_name] = program_entry
            logger.debug(f"Added program-mcp server with program_path: {program_path}")
        else:
            logger.warning(f"Program server '{program_server_name}' already exists; not overriding. Set override_program=True to override.")
    
    # Add program-dft-mcp server entry if requested
    if "program-dft-mcp" in init_servers:
        if program_dft_input_path is None:
            raise ValueError("program_dft_input_path must be provided when 'program-dft-mcp' is in init_servers")
        
        program_dft_entry = _program_dft_server_entry(
            input_path=program_dft_input_path,
            timeout=program_dft_timeout,
        )
        
        program_dft_server_name = "program-dft"
        if override_program_dft or program_dft_server_name not in config["mcpServers"]:
            config["mcpServers"][program_dft_server_name] = program_dft_entry
            logger.debug(f"Added program-dft-mcp server with input_path: {program_dft_input_path}")
        else:
            logger.warning(f"Program-dft server '{program_dft_server_name}' already exists; not overriding. Set override_program_dft=True to override.")
    
    return config

def ensure_parents_exist(target_dir: str, relative_paths: list[str]) -> None:
    """
    Pre-create directories so 'save_file' won't fail if parent dirs are missing.
    """
    root = Path(target_dir).expanduser().resolve()
    for rel in relative_paths:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)