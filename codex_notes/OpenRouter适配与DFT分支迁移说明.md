# OpenRouter 适配与 DFT 分支迁移说明

本文总结当前工作分支最终版本中的 OpenRouter 相关修改，供另一个 Codex 会话在 DFT branch 中复刻。迁移时应先只读检查 DFT branch 的对应文件和调用链，因为文件名、入口脚本或局部实现可能不同，不应机械覆盖整份文件。

## 最终目标

- 用户只需在项目根目录 `.env` 中提供 `OPENROUTER_API_KEY`，即可运行普通文本 LLM 调用和 MCP-enabled LLM 调用。
- 普通文本调用默认使用 OpenRouter。
- MCP 调用默认使用 OpenRouter，但保留作者内部 provider 的原有分支。
- 原有 `send_chat_yidong.py` 和相关 provider 类不删除，避免破坏作者自己的运行环境。
- 对外文档优先使用 OpenRouter 作为可执行示例；自定义 provider 使用 `yourAPI` 作为通用占位符。
- 不在代码、测试、README 或日志中写入、展示 API key。

## 1. 配置与依赖前提

### `config.py`

当前项目原本已经包含：

```python
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
```

并通过 `python-dotenv` 加载项目根目录 `.env`。迁移 DFT branch 时先确认这一点；若 DFT branch 已有同样配置，不要重复实现。

用户配置方式：

```bash
OPENROUTER_API_KEY=your_key_here
```

不要读取或打印真实 key。

### `requirements.txt`

当前项目原本已有：

```text
openai==1.107.2
python-dotenv==1.1.1
Langchain_openai==0.3.33
```

因此新增适配器没有引入额外依赖。DFT branch 迁移时应核对对应依赖是否存在。

## 2. 新增纯文本 OpenRouter 适配器

新增文件：

```text
utils/send_chat_openrouter.py
```

公开接口：

```python
def send_chat_openrouter(
    user_model: str,
    role_prompt: str,
    user_prompt: str,
    temperature: float = None,
    top_p: float = None,
    max_tokens: int = None,
    iscaltoken: bool = False,
) -> Tuple[str, int, Optional[Dict]]:
    ...
```

关键实现约定：

- 使用 OpenAI-compatible SDK：

  ```python
  OpenAI(
      base_url="https://openrouter.ai/api/v1",
      api_key=OPENROUTER_API_KEY,
  )
  ```

- system prompt 和 user prompt 分别发送为 `system`、`user` message。
- `temperature`、`top_p`、`max_tokens` 为 `None` 时不传给 SDK。
- 返回格式与原有低层适配器兼容：

  ```python
  (response_text, status_code, token_dict)
  ```

- `iscaltoken=True` 时返回：

  ```python
  {
      "input_tokens": ...,
      "output_tokens": ...,
      "total": ...,
  }
  ```

- provider 未返回 usage 时，token 数量安全回退为 0。
- key 缺失时返回清晰的 401 配置错误，但不展示 key。
- API/网络异常时保留异常对象提供的 HTTP 状态码；无状态码时使用本地 500。这里的 500 可能只是本地包装的网络错误，并不一定是 OpenRouter 返回的 HTTP 500。
- 当前实现为非流式调用，足以兼容现有上层调用链。

## 3. 接入普通文本调用链

修改对应的 agent 基类文件；main 分支中为：

```text
src/QMBagents.py
```

### 导入适配器

```python
from utils import MCP_toolbox, run_program, code_editor, send_chat_openrouter, send_chat_yidong
```

### 修改 `_send_chat`

默认 provider 改为 OpenRouter：

```python
def _send_chat(
    self,
    user_model: str,
    user_prompt: str,
    temperature: float = None,
    top_p: float = None,
    api_type: str = "openrouter",
    iscaltoken: bool = False,
):
```

路由逻辑：

```python
if api_type == "openrouter":
    response_text, status_code, token_dict = (
        send_chat_openrouter.send_chat_openrouter(
            user_model=model_name,
            role_prompt=QMB_author_prompt,
            user_prompt=user_prompt,
            temperature=temp_set,
            top_p=top_p_set,
            iscaltoken=iscaltoken,
        )
    )
    ...
    return response_text, token_dict
elif api_type == "yidong":
    # 原分支完整保留
    ...
else:
    raise ValueError(f"Unsupported API provider: {api_type}")
```

注意事项：

- `_send_chat_robust` 无需逐个修改；它调用 `_send_chat` 时会自然使用新的默认 provider。
- 保留原 provider 分支，不重写或删除旧适配器。
- 删除不再需要的、直接在 `QMBagents.py` 内构造 `OpenAI` client 的旧内嵌逻辑；统一经过 `send_chat_openrouter.py`。
- 如果 DFT branch 的 agent 基类文件名不同，应定位 `_send_chat` 的真实定义后做等价修改。

## 4. 接入 MCP 调用链

修改：

```text
utils/MCP_toolbox.py
```

### provider 配置

在导入 `config` 后增加：

```python
# MCP LLM provider. Override with MCP_LLM_PROVIDER=openrouter|yidong in .env.
MCP_LLM_PROVIDER = os.getenv("MCP_LLM_PROVIDER", "openrouter").strip().lower()
```

默认值必须是 `openrouter`。

### 动态 MCP 路由

项目已经有 `ChatOpenRouter(ChatOpenAI)`，其 base URL 为：

```text
https://openrouter.ai/api/v1
```

在 `send_chat_through_mcp_dynamic()` 创建 LLM 的位置加入 provider 判断：

```python
if MCP_LLM_PROVIDER == "openrouter":
    llm = ChatOpenRouter(
        model=user_model,
        temperature=model_temp,
    )
elif MCP_LLM_PROVIDER == "yidong":
    # 原有 qwen/openai/anthropic 三类 Yidong 路由完整保留
    ...
else:
    raise ValueError(
        f"Unsupported MCP LLM provider: {MCP_LLM_PROVIDER}. "
        "Expected 'openrouter' or 'yidong'."
    )
```

注意：只修改普通 `_send_chat` 不足以跑完整工作流，因为 Author/Retrieve/Repair 等阶段会调用 `_send_chat_through_mcp_dynamic()`。DFT branch 也必须检查其完整工作流是否走 MCP。

缺少 Yidong key 不会影响默认 OpenRouter 路径：导入时这些环境变量只是 `None`，只要不选择或实例化 Yidong provider，就不会进入其 key 使用逻辑。

## 5. 真实 OpenRouter 测试脚本

新增：

```text
tests/test_send_chat_openrouter.py
```

该文件不是 mock 单元测试，而是用户可直接执行的真实连通性测试：

```bash
source .venv/bin/activate
python tests/test_send_chat_openrouter.py
```

最终版本特征：

- 直接导入并调用：

  ```python
  from utils.send_chat_openrouter import send_chat_openrouter
  ```

- 默认测试模型：

  ```python
  DEFAULT_TEST_MODEL = "qwen/qwen3-max"
  ```

- 可通过环境变量覆盖：

  ```bash
  OPENROUTER_TEST_MODEL="provider/model-name" \
      python tests/test_send_chat_openrouter.py
  ```

- 发送一次真实、低 token 请求，`max_tokens=32`。
- 打印模型、响应和 token usage，但绝不打印 API key。
- 非 200、空响应时退出码为 1；成功时退出码为 0。

## 6. `run_sh/` 模型名迁移

OpenRouter 需要 `provider/model` 格式的模型 ID。main 分支最终把相关脚本中的实际模型统一改为：

```bash
model_author="qwen/qwen3-max"
```

存在 `model_author_judge` 的脚本使用：

```bash
model_author_judge="qwen/qwen3-max" # or try "google/gemini-3-flash-preview" for better performance
```

仅 `model_author_judge` 后添加 Gemini 3 可选备注，其他模型变量不加该备注。

main 分支修改过的脚本包括：

- `run_task_workflow_compprog_v2.sh`
- `run_task_workflow_compprog_continue.sh`
- `run_task_workflow_compprog_v2_same2_TTS.sh`
- `run_task_workflow_compprog_v2_same2_TTS_batch.sh`
- `run_task_workflow_compscit_v4.sh`
- `run_task_baseline_ReAct_v2.sh`
- `run_task_baseline_ReAct_continue.sh`
- `run_task_baseline_compscit_v4.sh`

DFT branch 的脚本名称或角色变量可能不同。迁移时应搜索所有实际赋值：

```bash
rg -n '^(model_[A-Za-z0-9_]+)=' run_sh/*.sh
```

只替换真正使用的 provider 模型名，不修改任务列表、路径、GPU、迭代次数和其他实验参数。

## 7. README 最终表述原则

README 保留原有的完整配置说明，不能为了突出 OpenRouter 而删除以下内容：

- `.env` 的作用；
- `config.py` 中添加自定义 provider key/config 的说明；
- 普通文本 adapter 的结构与 `_send_chat` 路由方式；
- MCP 中 LangChain-compatible wrapper 与动态路由方式；
- 使用前单独测试 API 的建议。

在原说明基础上增加 OpenRouter 内容：

- `send_chat_openrouter.py` 是项目提供的、其他用户可直接使用的完整参考适配器。
- `.env` 中加入 `OPENROUTER_API_KEY` 后即可使用。
- `_send_chat` 当前默认 provider 是 OpenRouter。
- `MCP_toolbox.py` 顶部的 `MCP_LLM_PROVIDER` 默认是 `openrouter`。
- README 中可执行示例优先展示 OpenRouter。
- 自定义 provider 统一使用 `yourAPI` 占位，例如：

  ```text
  utils/send_chat_yourAPI.py
  api_type="yourAPI"
  ChatYourAPI
  MCP_LLM_PROVIDER == "yourAPI"
  ```

- 可以简单提及 `send_chat_yidong.py` 是作者内部环境使用的模块，其服务和凭据通常不向其他用户开放；外部用户应直接使用 OpenRouter 或实现自己的 `yourAPI`。
- 不要把 Yidong 当作面向外部用户的主要教程示例。

README 的 OpenRouter 直接调用示例应类似：

```python
from utils import send_chat_openrouter

response, status_code, token_dict = send_chat_openrouter.send_chat_openrouter(
    user_model="qwen/qwen3-max",
    role_prompt="You are a helpful assistant.",
    user_prompt="Who are you?",
    temperature=0.1,
    top_p=0.9,
    iscaltoken=True,
)
```

## 8. 网络代理注意事项

当前服务器某些终端设置了失效的本地代理，例如 `HTTPS_PROXY` 指向远程服务器上未监听的 `127.0.0.1` 端口。这会导致 OpenAI SDK 报：

```text
OpenRouter request failed: Connection error.
```

若服务器可直接访问 OpenRouter，可为单次命令取消代理环境变量：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    python tests/test_send_chat_openrouter.py
```

运行完整脚本时同理。由于项目大量使用 `../` 相对路径，应从 `run_sh/` 目录启动：

```bash
cd /path/to/project/run_sh
source ../.venv/bin/activate

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    bash ./run_task_workflow_compprog_v2.sh
```

DFT branch 应替换为其实际入口脚本名。虚拟环境决定 Python 依赖，但不决定网络是否可达。

## 9. DFT branch 建议执行顺序

1. 只读检查 DFT branch 的 `AGENTS.md`、README、`config.py`、requirements、agent 基类、`MCP_toolbox.py`、`run_sh/` 和现有 tests。
2. 找出普通文本 `_send_chat` 的真实定义及所有 provider 分支。
3. 找出完整 DFT 工作流中所有 `_send_chat_through_mcp_dynamic` 调用。
4. 确认 `OPENROUTER_API_KEY` 与所需依赖已存在。
5. 新增 `utils/send_chat_openrouter.py`。
6. 将普通 `_send_chat` 默认路由接到 OpenRouter，同时保留原 provider。
7. 给 `MCP_toolbox.py` 增加默认 OpenRouter 的 `MCP_LLM_PROVIDER` 路由，同时保留原 provider。
8. 新增真实低 token 连通性测试。
9. 将 DFT 相关运行脚本的实际模型名改为有效 OpenRouter ID。
10. 按上述原则更新 README，保留完整旧配置说明并补充 OpenRouter 快速路径。
11. 运行离线检查，再运行一次真实 OpenRouter 测试；未经用户授权不要直接启动耗时、昂贵的完整 DFT 工作流。

## 10. 验证清单

迁移完成后至少检查：

```bash
# Python 语法（也可用只读 ast.parse 避免生成 __pycache__）
python -m py_compile \
    utils/send_chat_openrouter.py \
    path/to/agent_base.py \
    utils/MCP_toolbox.py \
    tests/test_send_chat_openrouter.py

# Shell 语法
bash -n run_sh/<modified-script-1>.sh run_sh/<modified-script-2>.sh

# 格式检查
git diff --check

# 真实低 token API 测试
python tests/test_send_chat_openrouter.py
```

还应确认：

- OpenRouter 普通文本调用返回非空文本和正确 token 字典；
- MCP OpenRouter wrapper 能构造成功；
- 默认普通 provider 为 `openrouter`；
- 默认 MCP provider 为 `openrouter`；
- 原 provider 分支仍存在；
- 不配置内部 provider keys 时，默认 OpenRouter 路径仍能正常导入和运行；
- 所有运行脚本使用有效的 OpenRouter `provider/model` ID；
- README 没有泄露任何真实 key。

## 11. 当前工作分支状态提示

生成本文时，当前分支为：

```text
PRXI_review1_cleanup_nodata_public
```

当前提交：

```text
8f6b1620ed update: release prepare
```

工作树当时为干净状态，说明上述最新版改动已经包含在当前版本中。迁移 DFT branch 时以实际文件内容为准，不应假设两个分支逐行一致。
