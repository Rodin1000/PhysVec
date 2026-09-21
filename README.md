<h1 align="center">PhysVEC for Density Functional Theory (ORCA)</h1>

Welcome to the **DFT** release of PhysVEC — the same verifiable, multi-agent framework used in **QMP-Bench** quantum many-body benchmarks, adapted for **ORCA** density-functional and quantum-chemistry calculations.

Recent advances in LLM-driven scientific agents are prone to hallucination. PhysVEC addresses this with an **Author** agent plus **programming** and **integration** verifiers that check syntax, execution, and structured scientific outputs, with human-auditable logs at each step. Within the broader **QMP-Bench** benchmark (100 tasks from 21 papers), the DFT portion is implemented here under **`dft_qc`** (26 molecular quantum-chemistry tasks), all using **ORCA**. This repository branch ships the DFT implementation; the **main** PhysVEC branch covers **dmrg**, **nqs**, and **qsim**.

# Overview

## PhysVEC framework (DFT)

<p align="center">
  <img src="workflow_compare_1_a.png" alt="Workflow comparison" width="70%" />
</p>

- **Cooperative multi-agent design.** PhysVEC coordinates an **Author** agent, a **Programming verifier** (unit tests on ORCA input blocks), and an **Integration verifier** (end-to-end ORCA runs and result checks).
- **Author agent.** Reads the task (paper figure/table) and produces ORCA `.inp` inputs, then splits them into a small **repository** of block files (`keywords_*.inp`, `geometryblocks_*.inp`, `inputblocks_*.inp`, …).
- **Programming verifier (`unittest`).** Generates and runs library-backed verifier scripts in sandboxes; checks block-level correctness and ORCA execution.
- **Integration verifier (`integtest`).** Runs the full reproduced repository through ORCA and evaluates outputs against task rubrics.
- **Repair loop.** When tests fail, **`coderepair`** agents suggest fixes and the Author rewrites code until tests pass or iteration limits are reached.

Scientific-test drivers (`task_workflow_compscit_v4.py`) used on the QMB branch are **not included** on this DFT branch; programming + integration tests are the primary verification path here.

## DFT benchmark topics

<p align="center">
  <img src="workflow_compare_1_b.png" alt="Workflow comparison" width="70%" />
</p>

| Topic                | Package | Tasks (this repo) | Typical outputs                          |
| -------------------- | ------- | ----------------- | ---------------------------------------- |
| **`dft_qc`** | ORCA    | 26 JSON tasks     | XAS/XES, TDDFT, geo opt, MO diagrams, … |

Each task JSON under `Paper_dataset/<topic>/tasks/` points to a source article (`.tex` under `Paper_dataset/<topic>/…`) and a rubrics file. The agent must **reproduce the published figure/table** described in `User_requests`.

# Mapping to the QMB (main) codebase

If you know the main-branch layout, use this table:

| QMB / main branch                                   | DFT branch (this repo)                                                                                   |
| --------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| [`QMBagents.py`](src/QMBagents.py) → `QMBagent` | [`DFT_agents.py`](src/DFT_agents.py) → `DFT_agent`                                                   |
| [`repair_agents.py`](src/repair_agents.py)         | [`repair_agents_dft.py`](src/repair_agents_dft.py)                                                      |
| [`ReAct_agents.py`](src/ReAct_agents.py)           | [`ReAct_dft_agents.py`](src/ReAct_dft_agents.py)                                                        |
| `task_workflow_compprog_v2.py`                    | [`task_workflow_compprog_dft_v2.py`](src/task_workflow_compprog_dft_v2.py)                              |
| `task_workflow_compscit_v4.py`                    | *not shipped on DFT branch*                                                                            |
| Author step (embedded in compprog)                  | [`author_dft_v4.py`](src/author_dft_v4.py)                                                              |
| Programming tests                                   | [`unittest_dft_v9.py`](src/unittest_dft_v9.py)                                                          |
| Integration tests                                   | [`integtest_dft_v9.py`](src/integtest_dft_v9.py)                                                        |
| Repair step                                         | [`coderepair_dft_v7.py`](src/coderepair_dft_v7.py)                                                      |
| ReAct baseline                                      | [`task_baseline_ReAct_dft_v2.py`](src/task_baseline_ReAct_dft_v2.py)                                    |
| `prompts/general/`                                | [`prompts/general_dft/`](prompts/general_dft/) + topic packs e.g. [`prompts/dft_qc/`](prompts/dft_qc/) |
| `program-mcp` (Python/Julia)                      | [`program-dft-mcp`](MCP_servers/program-dft-mcp/) (`run_orca`)                                        |
| `CodeVerifier_library_<topic>`                    | `CodeVerifier_library_auto/CodeVerifier_library_<topic>/`                                              |

### Agent classes in `DFT_agents.py`

[`DFT_agents.py`](src/DFT_agents.py) mirrors [`QMBagents.py`](src/QMBagents.py): shared LLM/MCP plumbing plus role-specific subclasses:

| Class               | Role                                                                                 |
| ------------------- | ------------------------------------------------------------------------------------ |
| `DFT_agent`       | Base: prompts,`_send_chat`, `_send_chat_through_mcp_dynamic`, filesystem helpers |
| `PaperSummerizer` | Paper/task understanding                                                             |
| `CodeGenerator`   | ORCA input generation (MCP + RAG)                                                    |
| `RepoGenerator`   | Split monolithic`.inp` into block repository + README                              |
| `CodeVerifier`    | Unit-test verifier generation/execution                                              |
| `CodeIntegrator`  | Integration-test orchestration                                                       |
| `RubricsGrader`   | Rubric-based scoring                                                                 |

[`repair_agents_dft.py`](src/repair_agents_dft.py) defines `Code_repairer` for the repair loop.

# Code structure

## User interface

Executable entry points are **shell scripts** so you can set paths, tasks, and models in one place.

### Batch workflows — [`run_sh/`](run_sh/)

Run from `run_sh/` (scripts call `python ../src/...`):

| Script                                                                         | Purpose                                                                                                                                       |
| ------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| [`run_task_workflow_compprog_v2.sh`](run_sh/run_task_workflow_compprog_v2.sh) | **PhysVEC programming phase:** Author → loop (`unittest` + `integtest` + postprocess + `coderepair`). Default topic: `dft_qc`. |
| [`run_task_baseline_ReAct_v2.sh`](run_sh/run_task_baseline_ReAct_v2.sh)       | **ReAct baseline** with optional RAG repair.                                                                                            |
| [`run_task_workflow_compscit_v4.sh`](run_sh/run_task_workflow_compscit_v4.sh) | Placeholder — requires`task_workflow_compscit_v4.py` from main branch.                                                                     |
| [`run_task_baseline_compscit_v4.sh`](run_sh/run_task_baseline_compscit_v4.sh) | Placeholder — requires`task_baseline_compscit_v4.py` from main branch.                                                                     |

### Single-task debugging — [`run/`](run/)

| Script                                                    | Python driver                                   | Purpose                                              |
| --------------------------------------------------------- | ----------------------------------------------- | ---------------------------------------------------- |
| [`run_author_dft_v4.sh`](run/run_author_dft_v4.sh)       | [`author_dft_v4.py`](src/author_dft_v4.py)     | Author only: plan → code → repo → rule refinement |
| [`run_unittest_dft.sh`](run/run_unittest_dft.sh)         | [`unittest_dft_v9.py`](src/unittest_dft_v9.py) | Unit tests on one repo                               |
| [`run_codeverifier_dft.sh`](run/run_codeverifier_dft.sh) | `codeverifier_v1_dft.py` *(if present)*     | Build/extend verifier library                        |

Version suffixes (`_v4`, `_v9`, `_v7`) follow the same convention as the QMB repo: **higher number = current driver** for that step.

## Pipeline construction — [`src/`](src/)

### PhysVEC programming workflow

[`task_workflow_compprog_dft_v2.py`](src/task_workflow_compprog_dft_v2.py) batches:

1. **Author** — [`author_dft_v4.py`](src/author_dft_v4.py)
2. **Loop** until pass or `max_workflow_iter`:
   - [`unittest_dft_v9.py`](src/unittest_dft_v9.py) + [`unittest_postprocess.py`](src/unittest_postprocess.py)
   - [`integtest_dft_v9.py`](src/integtest_dft_v9.py)
   - [`coderepair_dft_v7.py`](src/coderepair_dft_v7.py) using [`repair_agents_dft.py`](src/repair_agents_dft.py)

### Baseline

[`task_baseline_ReAct_dft_v2.py`](src/task_baseline_ReAct_dft_v2.py) drives [`ReAct_dft_agents.py`](src/ReAct_dft_agents.py) (generate → execute ORCA → suggest → repair).

## Benchmark dataset — [`Paper_dataset/`](Paper_dataset/)

```
Paper_dataset/
├── dft_qc/
│   ├── tasks/              # task_dft_qc_*.json
│   └── quantum_chemistry/  # source .tex (and helpers)
```

Example task fields: `pdf_name`, `subplot_name`, `User_requests`, `rubrics_name`. Source papers live beside each topic (not a separate `sources/` tree like `dmrg` on main).

## Sandboxes and verifier library

| Path                                                                                                                | Role                                     |
| ------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| [`CodeVerifier_sandbox/`](CodeVerifier_sandbox/)                                                                   | Per-task unit-test sandboxes             |
| [`CodeIntegrator_sandbox/`](CodeIntegrator_sandbox/)                                                               | Integration-test sandboxes               |
| [`CodeVerifier_library_auto/CodeVerifier_library_dft_qc/`](CodeVerifier_library_auto/CodeVerifier_library_dft_qc/) | Verified ORCA snippet library (`.inp`) |
| [`authority_library/orca-manual/`](authority_library/orca-manual/)                                                 | ORCA manual text for RAG / retrieve-mcp  |

To add verifier snippets for `dft_qc`, create a folder such as `CodeVerifier_library_auto/CodeVerifier_library_dft_qc/<Type>_elements/`, add a [`guide.json`](guide.json) (see root example: `topic`, `types`, `elements`), then run the codeverifier builder when `src/codeverifier_v1_dft.py` is available. Verified scripts land in `verifier_code/`.

# Configurations

For additional configuration guidance, see the related sections in the main branch README.

## ORCA

Add the absolute path to your installed ORCA executable to the project-root `.env` file:

```dotenv
ORCA_EXECUTABLE="/absolute/path/to/orca"
```

## LLM API calling

### Pure text

Use `_send_chat` / `_send_chat_robust` in [`DFT_agents.py`](src/DFT_agents.py). We provide [`utils/send_chat_openrouter.py`](utils/send_chat_openrouter.py) as a ready-to-use adapter. After adding `OPENROUTER_API_KEY` to the project-root `.env`, `_send_chat` uses OpenRouter by default. For your own provider, add an adapter such as `utils/send_chat_yourAPI.py` and extend the `api_type` routing. [`utils/send_chat_yidong.py`](utils/send_chat_yidong.py) is retained for our internal environment, but its service and credentials are not generally available to other users.

### MCP-enabled calls

`_send_chat_through_mcp_dynamic` in [`DFT_agents.py`](src/DFT_agents.py) delegates to [`send_chat_through_mcp_dynamic`](utils/MCP_toolbox.py) (LangChain chat models + MCP tools). `MCP_LLM_PROVIDER` near the top of this file defaults to `openrouter` and selects the LangChain-compatible `ChatOpenRouter` wrapper, using the same `OPENROUTER_API_KEY`. For another provider, add a wrapper such as `ChatYourAPI` and extend the provider routing with its URL, model names, and credentials. The internal provider wrappers are retained for reproducibility.

DFT-specific helpers: [`utils/code_editor_dft.py`](utils/code_editor_dft.py), [`utils/run_program.py`](utils/run_program.py).

## MCP servers — [`MCP_servers/`](MCP_servers/)

| Server                    | DFT usage                                                                      |
| ------------------------- | ------------------------------------------------------------------------------ |
| **filesystem-mcp**  | Read/write repos, logs, reference projects                                     |
| **retrieve-mcp**    | RAG over`authority_library/` (e.g. ORCA manual)                              |
| **program-dft-mcp** | Execute ORCA`.inp` via `run_orca` (replaces **program-mcp** on main) |

From the project root, install each server in the **same** project venv (do not create separate venvs per server):

```bash
python -m pip install -e MCP_servers/filesystem-mcp
python -m pip install -e MCP_servers/retrieve-mcp
python -m pip install -e MCP_servers/program-dft-mcp
```

Configure **`ORCA_EXECUTABLE`** in the project-root `.env` file as described in [ORCA configuration](#orca). See [`MCP_servers/program-dft-mcp/README.md`](MCP_servers/program-dft-mcp/README.md) for the `run_orca` tool API.

### Local knowledge base (retrieve-mcp)

1. Libraries live under [`authority_library/`](authority_library/) (e.g. `orca-manual/`).
2. `cd MCP_servers/retrieve-mcp`
3. Run `python download_embedding_model.py` to download the embedding model.
4. After downloading, run the following from the project root (Linux/WSL) to create the runtime compatibility link:

   ```bash
   ln -s MCP_servers/retrieve-mcp/models/embedding_model embedding_model
   ```

   This link lets the vector store builder and runtime retrieval service share the same model files without downloading them twice. The root `embedding_model` path must not already exist when creating the link.

5. From `MCP_servers/retrieve-mcp`, run `build_vector_store.sh` / `check_vector_store.sh` / `delete_vector_store.sh` as on the main branch.

# Setup guideline

## Step 1: Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**ORCA** must be installed, with `ORCA_EXECUTABLE` configured in the project-root `.env` file as described in [ORCA configuration](#orca).

**Julia** ([`Project.toml`](Project.toml) / [`Manifest.toml`](Manifest.toml)) is **not required** for DFT/ORCA tasks on this branch.

## Step 2: MCP servers

Install **filesystem-mcp**, **retrieve-mcp**, and **program-dft-mcp** as in [MCP servers](#mcp-servers--mcp_servers). Build the vector store for `orca-manual` (or other libraries you use).

## Step 3: LLM API configuration

Store keys in a root `.env` file (keep `.env` in `.gitignore`). Add keys or configuration for custom providers in [`config.py`](config.py). For the provided OpenRouter adapter, add:

```bash
OPENROUTER_API_KEY=your_key_here
```

OpenRouter is the default for both text and MCP calls. Use valid OpenRouter model IDs such as `qwen/qwen3-max` in the run scripts.

1. **Text API:** to use another provider, add `utils/send_chat_yourAPI.py` with the same function signature and `(response_text, status_code, token_dict)` return shape, then add an `api_type="yourAPI"` branch in `_send_chat` in [`DFT_agents.py`](src/DFT_agents.py).
2. **MCP API:** define a LangChain-compatible `ChatYourAPI` wrapper and add an `MCP_LLM_PROVIDER == "yourAPI"` branch in [`utils/MCP_toolbox.py`](utils/MCP_toolbox.py). Configure its URL, model names, and credentials; the other method definitions do not need to change.

Test the API separately before running the full pipeline. OpenRouter example:

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
print(response, status_code, token_dict)
```

You can also run the included real connectivity test (one low-token request):

```bash
python tests/test_send_chat_openrouter.py
```

## Step 4: Run the pipeline

1. Edit task list, models, and directories in [`run_sh/run_task_workflow_compprog_v2.sh`](run_sh/run_task_workflow_compprog_v2.sh) (or a [`run/`](run/) script for a single task).
2. From `run_sh/`: `bash run_task_workflow_compprog_v2.sh`

Typical knobs:

- `topic` — `dft_qc`
- `task_list` / `isall`
- `result_dir`, `output_repo_dir`, sandbox roots
- `model_step_*` — per-role LLM IDs
- `max_workflow_iter`, `max_iter_refinecode_rules`, `max_iter_retry`

Outputs appear under `results/…/<topic>/task_<name>/` with `_check` / `_retry*` / `repo_*` directories and `report_*.jsonl` files.

## Step 5 (optional): Verifier library

Pre-built examples: `CodeVerifier_library_auto/CodeVerifier_library_dft_qc/verifier_code/*.inp`.

To add types (e.g. new XAS workflow blocks):

1. Create `CodeVerifier_library_auto/CodeVerifier_library_dft_qc/<name>_elements/`.
2. Add `guide.json` (`topic`, `types`, `elements`, `element_notes`) — see root [`guide.json`](guide.json).
3. Run [`run/run_codeverifier_dft.sh`](run/run_codeverifier_dft.sh) when `src/codeverifier_v1_dft.py` is available on your checkout.

---

For the full PhysVEC paper, QMP-Bench composition (dmrg / nqs / qsim / DFT), and scientific-test methodology, see the **main** repository branch and the Methods section of the paper. This README documents the **DFT/ORCA** codebase on the current branch.
