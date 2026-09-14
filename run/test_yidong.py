from openai import OpenAI
import time
import random
from pathlib import Path
import json
import base64
from typing import Optional, Dict, List, Tuple
import fitz
import re
from langchain_openai import ChatOpenAI
import os
import sys
import asyncio
import ast
import shutil
from decimal import Decimal, InvalidOperation
from mcp_use import MCPAgent, MCPClient
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# Import from this project
from config import OPENROUTER_API_KEY, YIDONG_API_KEY, YIDONG_API_KEY_2, YIDONG_API_KEY_3, YIDONG_API_KEY_1
from config import get_model_settings
from utils import MCP_toolbox, run_program, code_editor, send_chat_yidong, send_chat_yidong_patched


# response, status_code, token_dict = send_chat_yidong.send_chat_diverse_model(user_model="claude-3-haiku-20240307", role_prompt="You are a helpful assistant.", user_prompt="Who are you?", temperature=0.1, top_p=0.9, iscaltoken=True, isstream=False)

# print(response)
# print(status_code)
# print(token_dict)

url = "https://yinli.one/v1/chat/completions"

payload = {
    "model": "deepseek-v3",
    "messages": [
        {
            "role": "user",
            "content": "你好，测试下通不通"
        }
    ],
    "max_tokens": 50
}

headers = {
    "Content-Type": "application/json",
    "Authorization": YIDONG_API_KEY_2
}

response = requests.post(url, json=payload, headers=headers)

print("status:", response.status_code)
print(response.text)

# if __name__ == "__main__":
#     text, status, tokens = send_chat_yidong_patched.send_chat_diverse_model(
#         user_model="gemini-3-flash-preview-thinking",
#         role_prompt="",                      # 不需要 system prompt
#         user_prompt="你好，测试下通不通",
#         temperature=0.2,
#         max_tokens=200,
#         iscaltoken=True,
#         isstream=True                     # ⭐ 关键：先关流式
#     )

#     print("status:", status)
#     print("text:", text)
#     print("tokens:", tokens)


# url = "https://yinli.one/v1/messages"


# body = {
#     "model": "gemini-3-flash-preview",
#     "messages": [
#         {
#             "role": "user",
#             "content": "你好，测试下通不通"
#         }
#     ],
#     "max_tokens": 200
# }


# response = requests.post(
#     url=url,
#     json=body, 
#     headers={
#         "anthropic-version": "2023-06-01",
#         "Authorization": YIDONG_API_KEY_1
#     }
# )

# print("响应状态码:", response.status_code)  # 打印状态码，方便排查
# print("响应内容:", response.text)


# url = "https://yinli.one/v1/chat/completions"

# payload = {
#     "model": "deepseek-r1",
#     "messages": [
#         {
#             "role": "user",
#             "content": "你好，测试下通不通"
#         }
#     ],
#     "max_tokens": 50
# }

# headers = {
#     "Content-Type": "application/json",
#     "Authorization": YIDONG_API_KEY_2
# }

# response = requests.post(url, json=payload, headers=headers)

# print("status:", response.status_code)
# print(response.text)





# temperature = 0.1
# max_tokens = 200

# url = "https://yinli.one/v1/messages"
# body = {
#   "model": "grok-4-fast-reasoning",
#   "messages": [
#     {
#       "role": "user",
#       "content": "你好，测试下通不通"
#     }
#   ],
#   "max_tokens": max_tokens,
#   "temperature": temperature
# }
# response = requests.request("POST", url, data = json.dumps(body), headers = {
#   "Content-Type": "application/json", 
#   "anthropic-version": "2023-06-01", 
#   "Authorization": YIDONG_API_KEY
# })

# print(body)
# print(response.status_code)
# print(response.text)
# text = response.text
# text = json.loads(text)
# print(text["usage"]["input_tokens"])
# print(text["usage"]["output_tokens"])


# content = """Your task is to generate executable and practical codes to reproduce results of a research paper, according to given informaiton. 
# During this process, you should strive to ensure that the code faithfully represents the task requirements while maintaining its executability. You should not provide a incomplete code. 
# You are requested to use MCP tools to look up into local library for related information. 

# -----------------------------------------------------------------------------------------------
# [Input]:
# 1. Plan_info: You will be given a task plan, whose name may be 'Plan_info', ...The task plan aims to instruct you to reproduce a subplot/table. It typically contains:
#     - "subplot_name": The user defined subplot that you will focus on. May be Fig. 1(c), Fig. 1(a)-inset, ...Insets should be regarded as seperate subplots. If you cannot find such subplot in the paper, this term is 'Subplot not found', and all items below is 'None'. 
#     - "User_requests": User's additional instructions for generating the subplot. Including specifying several items, specifying numerical methods, ...
#     - "plan": a detailed plan containing information about how to generate code, specific values that should be implemented...


# ----------------------------------------------------------------------------------------------
# [MCP usage]:
# Before code generation process, please use MCP to get more related information/guides. 
# 1. You may use MCP server: 'filesystem' to get information from local libraries. Please follow the searching strategy below:
#     - IMPORTANT: The summary file is in the reference project ("libs"), NOT necessarily in the filesystem-mcp working directory.
#    - First call filesystem-mcp `get_reference_projects` and confirm reference "libs" exists
#    - Then call filesystem-mcp `list_reference_directory(reference_name="libs")` to list available files
#    - Use filesystem-mcp `read_reference_file(reference_name="libs", file_path="<summary json relative path>")` to read the summary JSON
#         - file_path MUST be the exact RELATIVE path returned by list_reference_directory (no `..\\`, no absolute paths)
#     - Tool arguments must be a JSON object, not a JSON string. Never wrap the arguments object in quotes.


# -----------------------------------------------------------------------------------------------
# [Instructions]:
# You will generate a code file (py, julia, ) according to the task plan. 
# The code should be consist of a set of functions. You MUST STRICTLY FOLLOW the rules below. 


# "Below is a complete, self‑contained and executable plan to reproduce Fig. 4(a) (left panel) of the paper 'Solving the Quantum Many‑Body Problem with Artificial Neural Networks' using ONLY this description and without requiring the original paper.\n\n-----------------------------------------------\n1. WHAT IS PLOTTED IN FIG. 4(a)\n-----------------------------------------------\nThe subplot shows the time evolution of the transverse magnetization <σ_x(t)> of a 1D transverse‑field Ising (TFI) chain with periodic boundary conditions (PBC) after a quantum quench h_i → h_f.\nThe paper considers:\n• 1D TFI Hamiltonian with PBC:\n  H = - h Σ_i σ_i^x - Σ_i σ_i^z σ_{i+1}^z\n• System size for dynamics: N = 40 spins\n• Two quenches: (requested by user)\n  (A) h_i = 4   → h_f = 2\n  (B) h_i = 1/2 → h_f = 1\n• Observable to plot:\n  m_x(t) = (1/N) Σ_i <σ_i^x(t)>\n• Time window not explicitly given in the text; from the figure one observes evolution up to t ~ 6–8 (moderate guess). Use t ∈ [0, 8] with Δt = 0.05.\n• Use a Restricted Boltzmann Machine (RBM/NQS) ansatz with hidden‑unit density α = 4.\n• Use time‑dependent variational Monte Carlo (t‑VMC) as defined in the Supplement.\n• Only show NQS curves; ignore exact results.\n\n-----------------------------------------------\n2. REQUIRED EXTERNAL PACKAGES\n-----------------------------------------------\nYou MUST use:\n• NetKet 3 (neural‑network wavefunctions, VMC and t‑VMC)\nYou MAY also include (not for main computation):\n• NumPy (standard)\n• Matplotlib (just for plotting)\nNo ITensor/DMRG is used here because user asks to only reproduce NQS results.\n\n-----------------------------------------------\n3. PREPARATION: DEFINE EXACT MODEL IN NETKET\n-----------------------------------------------\n(1) Lattice:\n  Use NetKet.graph.Hypercube(n_dim=1, length=40, pbc=True)\n\n(2) Hilbert space:\n  Spin-1/2 degrees of freedom:\n  netket.hilbert.Spin(s=1/2, N=40)\n\n(3) Hamiltonian for propagation (final h_f):\n  For the TFI chain:\n    H = -h_f * Σ_i σ^x_i - Σ_i σ^z_i σ^z_{i+1}\n  Implement in NetKet using:\n    netket.operator.LocalOperator\n    or directly:\n    netket.operator.Ising(h=h_f, hilbert=hilbert, graph=graph)\n  where NetKet’s Ising operator uses the convention:\n    H = -Σ Z_i Z_j - h Σ X_i\n  (This matches exactly the paper.)\n\n-----------------------------------------------\n4. INITIAL STATE: GROUND STATE AT h_i USING STATIC VMC\n-----------------------------------------------\nFor each quench:\n• Construct Ising Hamiltonian with h = h_i.\n• Define RBM ansatz with density α = 4:\n    machine = nk.models.RBM(alpha=4)\n• Create variational state:\n    vstate = nk.vqs.MCState(sampler=nk.sampler.MetropolisLocal(...), model=machine, n_samples=2000)\n• Optimize using NetKet Stochastic Reconfiguration:\n    optimizer = nk.optimizer.Sgd(learning_rate=0.02)\n    driver = nk.VMC(hamiltonian=H_i, optimizer=optimizer, variational_state=vstate, preconditioner=nk.optimizer.SR(diag_shift=0.01))\n  Run for ~300–500 iterations until convergence of energy.\n• Save optimized parameters as initial parameters W(0).\n\n-----------------------------------------------\n5. TIME EVOLUTION (t-VMC) USING H_f\n-----------------------------------------------\nProcedure:\n• Create final Hamiltonian H_f with h = h_f.\n• Reuse trained variational state vstate (same RBM, same parameters).\n• Use NetKet’s time evolution driver:\n    driver = nk.TDVP(hamiltonian=H_f, variational_state=vstate, integrator=nk.dynamics.Euler(), preconditioner=nk.optimizer.SR(diag_shift=1e-3))\n• Time parameters:\n    t_max = 8.0\n    dt = 0.05\n    Nt = int(t_max / dt)\n• At each step:\n    driver.advance(dt)\n    compute m_x(t) = vstate.expectation(sigma_x_op)\n\nConstruct the σ_x observable:\n• For one site: sx = netket.operator.spin.sigma_x(0)\n• For full chain average:\n    sx_total = sum over sites netket.operator.spin.sigma_x(i)\n    m_x(t) = sx_total.expectation / N\n\n-----------------------------------------------\n6. RUN TWO SEPARATE SIMULATIONS\n-----------------------------------------------\nQuench 1:\n• h_i = 4\n• h_f = 2\n• Train ground state at h_i\n• Time evolve with H_f (h_f = 2)\n• Record curve m_x(t)\n\nQuench 2:\n• h_i = 1/2\n• h_f = 1\n• Train ground state at h_i\n• Time evolve with H_f (h_f = 1)\n• Record curve m_x(t)\n\n-----------------------------------------------\n7. OUTPUT FORMAT (PLOTTING)\n-----------------------------------------------\nThe subplot contains:\n• Time on horizontal axis: t ∈ [0, 8]\n• Vertical axis: transverse magnetization m_x(t)\n• Plot two solid lines:\n  – one for quench 4→2\n  – one for quench 1/2→1\nNo exact results are plotted.\n\n-----------------------------------------------\n8. NUMERICAL SETTINGS REQUIRED FOR REPRODUCIBILITY\n-----------------------------------------------\n• System size: N = 40 spins\n• Hidden‑unit density: α = 4 (machine = RBM(alpha=4))\n• Sampler: MetropolisLocal\n    number_of_samples_per_iteration = 2000\n    number_of_chains = 16\n• VMC optimization settings:\n    optimizer learning rate 0.02\n    SR diag_shift: 0.01\n    number of iterations: 300–500\n• Time evolution settings:\n    SR diag_shift: 1e‑3\n    integrator: Euler\n    dt = 0.05\n    total time = 8.0\n• Random seed:\n    Use seed = 123 for sampler, machine parameters, and optimization.\n\n-----------------------------------------------\n9. EXPECTED RESULTS\n-----------------------------------------------\n• Both quenches produce oscillatory behavior of m_x(t) around a mean value.\n• Larger quench (4→2) yields fast oscillations with larger amplitude.\n• Critical quench (1/2→1) produces slower oscillations.\nThese trends match Fig. 4(a) in the paper.\n\n-----------------------------------------------\n10. FINAL NOTE\n-----------------------------------------------\nAll steps above are sufficient to fully reproduce the neural‑network curves seen in Fig. 4(a) using only NetKet, without referring back to the paper. All model definitions, parameters, optimization procedures, and time‑evolution details are explicitly specified.\n"



# {
#   "summary": [
#     {
#       "query": "NetKet expectation value computation example",
#       "answer": "Expectation values in NetKet can be computed using MCState.expect or via low-level netket.jax.expect. MCState.expect supports chunking through the chunk_size argument to control memory. netket.jax.expect(params, σ, expected_fun, expected_fun_args=None, n_chains=None, chunk_size=None, in_axes=None) returns (value, Stats). Inputs: params are model parameters; σ are samples; expected_fun is the local estimator; expected_fun_args are additional differentiable args. Chunking and n_chains can be used to distribute computations."
#     },
#     {
#       "query": "NetKet Ising Hamiltonian construction",
#       "answer": "To build an Ising Hamiltonian: define a Hilbert space (e.g. hilbert = nk.hilbert.Spin(0.5, N, constraint=SumConstraint(0)) for fixed magnetization), define a graph (e.g. graph = nk.graph.Chain(N)), then construct the Hamiltonian: hamiltonian = nk.operator.Ising(hilbert, graph, h=1.0). This integrates seamlessly with samplers, variational states, and drivers."
#     },
#     {
#       "query": "NetKet MCState sampler options",
#       "answer": "NetKet samplers include MetropolisSampler, MetropolisSamplerNumpy, ExactSampler, ARDirectSampler, and ParallelTemperingSampler. MCState internally maintains sampler and sampler_state attributes. Sampling functions include sample() and sample_next(). Samplers are immutable; updated versions can be created via sampler.replace(). Sampler state objects are created with netket.sampler.init_state and passed to sampling routines."
#     },
#     {
#       "query": "NetKet RBM model definition example",
#       "answer": "RBM models are available via nk.models.RBM (real parameters by default). Other variants include RBMModPhase, RBMMultiVal, and RBMSymm. RBM models can be passed directly to variational states such as MCState. Construction example: model = nk.models.RBM(alpha=1). Used with vstates: vstate = nk.vqs.MCState(sampler, model)."
#     },
#     {
#       "query": "NetKet TDVP time evolution usage",
#       "answer": "Time evolution with TDVP is available via nk.experimental.TDVP or driver.TDVPSchmitt. It performs variational time evolution for MCState (t-VMC). Integrators support adaptive schemes and require specifying initial time t0, step size dt, and evolution time. TDVP is invoked with driver = nk.experimental.TDVP(hamiltonian, integrator=...). Then run(driver, vstate, T)."
#     }
#   ]
# }


# **Detailed Rules for Generating [NetKet] Codes**

# ====================================================================================================

# ## **CRITICAL: Function Format Requirements (MANDATORY)**

# **Every function MUST follow this exact format:**

# ```
# # Input dependency: ...(elements needed as input, from user or other functions. Format: "name"[brief description, data format])
# # Output results: ...(output variables of this function)
# function FUNCTION_NAME(parameters):
# ```

# **Key Requirements:**
# 1. **Function naming:** MUST be in form of `"tag_xxx"` (e.g., `hilbert_BoseHubbard1D`, `hamiltonian_XXZchain`)
# 2. **Type annotations:** MUST append type annotations (::T, e.g., ::Int, ::Float64) to ALL function INPUT parameters
# 3. **Common NetKet data types:**
#    - `g` (graph structure for a lattice system): `netket.graph.Lattice`
#    - `hi` (Hilbert space for a system): `netket.hilbert.Spin` (only if it is a spin system)
#    - `ha` (Hamiltonian): `netket.operator.LocalOperatorJax`
#    - `ma` (a variational wavefunction): `netket.models.RBMSymm` (only if it is created by RBMSymm method)
#    - `sa` (sampler): `netket.sampler.MetropolisLocal`
#    - `op` (optimizer): 'netket.optimizer.Sgd'/'netket.optimizer.Adam'/'netket.optimizer.SR'...
#    - `vs` (variational quantum state): `netket.vqs.MCState`
#    - `driv` (driver): `netket.driver.VMC`

# **CRITICAL:** You MUST STRICTLY FOLLOW this format for ALL functions.

# ====================================================================================================

# ## **Function Decomposition Principle**

# **You MUST avoid monolithic functions.** Decompose logic into smaller units using different tags (listed below).

# ====================================================================================================

# ## **Notation**

# **Function Categories:**
# - **ELEMENT functions:** Functions with tags {'hilbert', 'hamiltonian', 'observable', 'statemodel', 'compset', 'effector'}. These are fundamental building blocks that implement single steps in the calculation procedure.
# - **Task functions:** Functions with tags {'run', 'item', 'plot'}. These orchestrate ELEMENT functions to complete calculation tasks.
# - **Entry point:** the main function

# **Terminology:**
# - **"task"**: The whole reproduction process of the specified subplot/table
# - **"item"**: Individual elements in a subplot (e.g., curves, lines, point sets)

# **CRITICAL RULE:** **You are FORBIDDEN** to combine multiple tags for one function. For example, `"hilbert_hamiltonian_"` is **FORBIDDEN**.

# ====================================================================================================

# ## **Element function rules**

# ### **General rules for ALL element functions**

# 1. **Purpose:** Functions with ELEMENT tags aim to implement a SINGLE function/step in the whole calculation procedure

# 2. **You are REQUIRED** to create ALL tags in 'ELEMENT' (you cannot skip any)

# 3. **You are FORBIDDEN** to absorb some 'ELEMENT' function definitions into other ELEMENT function definitions

# 4. **You are STRICTLY FORBIDDEN** to call any other 'ELEMENT' functions in the definition of any function in 'ELEMENT'
#    - **Instead:** Define the required preceding variables as the input parameters of the function

# 5. **You are FORBIDDEN** to use global parameters or specific values directly
#   - **You are REQUIRED** to include every parameter used in each function definition in the input parameters of this function and DO NOT include specific values.

# 6. **Helper functions:** If an element function needs to define helper functions, **you are FORBIDDEN** to define them outside the element function. **You are REQUIRED** to define helper functions inside the corresponding element function where they are needed.


# ### **Detailed rules for each ELEMENT TAG:**

# #### **1. "hilbert"** - Lattice Site and Hilbert Space Definition
# - **Purpose:** Define the lattice type along with its Hilbert space for the targeted model (realizing "phy_model" in task plan)
# - **Examples:** spin 1/2 chain, ...
# - **Implementation:** Use `nk.graph.xxx` and `nk.hilbert.xxx` from NetKet
# - **Output results:** the graph object `g` and the hilbert space object `hi`
# - **Input dependency:** None (creates base objects)

# #### **2. "hamiltonian"** - Hamiltonian Definition
# - **Purpose:** Define the Hamiltonian of the model (realizing "phy_model" in task plan)
# - **Implementation:** Use `netket.operator.xxx` to create a Hamiltonian
# - **Input dependency:** Results from 'hilbert' function
# - **Output results:** the hamiltonian object `ha`

# #### **3. "statemodel"** - Variational State Format Definition
# - **Purpose:** Define the variational state format (model)
# - **Implementation:** Use `nk.models.RBMSymm`, `nk.models.RBM`, etc.
# - **Output results:** the model object `ma`
# - **Example:** Function that contains `nk.models.RBMSymm`, `nk.models.RBM`, ...

# #### **4. "compset"** - Sampler, Optimizer, and Variational State Setup
# - **Purpose:** Define the sampler, optimizer, and the variational quantum state used later
# - **Input dependency:** hilbert space object `hi`, graph object `g`, model object `ma`
# - **Output results:** sampler object `sa`, optimizer object `op`, variational quantum state object `vs`

# #### **5. "observable"** - Observable Definition
# - **Purpose:** Define observables needed in the research (according to "dependent" term in task plan)
# - **CRITICAL RULES:**
#   - **You are FORBIDDEN** to do VMC calculations directly in this function, you ONLY define the observables AFTER the calculation setup.
#   - **You are FORBIDDEN** to split a measurement into an 'observable' function and another 'measure' function
#   - **You are FORBIDDEN** to call former functions like 'hilbert', 'hamiltonian', 'effector', ...
#   - **You are REQUIRED** to wrap ALL commands in an 'observable' function for a single targeted observable
# - **Example:** Function that defines the correlation operator as "Cdag_i C_j"
# - **Note:** If multiple observables exist, each should correspond to a separate function

# #### **6. "effector"** - Main Calculation Operations
# - **Purpose:** Define and execute the main calculation (e.g., VMC_SR)
# - **CRITICAL RULES:**
#   - **You are FORBIDDEN** to name functions without any VMC operation as 'effector'. **You are FORBIDDEN** to name functions with calculations for observables or post-process as 'effector'.
#   - You first define a netket driver object `driv`, then use `driv.run(...)` method to execute the calculation
# - **Input dependency:** hamiltonian object `ha`, optimizer object `op`, variational quantum state object `vs`, and other parameters related to the calculation settings (n_samples, n_iter, ...)
# - **Output results:** a netket driver object `driv`


# ====================================================================================================

# ## **Task function rules**

# ### **1. "run"** - Complete Calculation for Single Parameter Set
# - **Purpose:** Implement calculation for a SINGLE set of inputs, DO NOT scan parameters here (Scanning process all in 'item' functions). 
# - **CRITICAL RULES:**
#   - **You are REQUIRED** to call 'hilbert'/'hamiltonian'/'observable'/'statemodel'/'compset'/'effector' tagged functions (AT LEAST ONE for each). You are REQUIRED** to implement a complete calculation procedure: you MUST BEGIN from constructing hilbert space, then call other element functions
#   - **You are FORBIDDEN** to define explicitly Hilbert space, Hamiltonian, observables, NetKet drivers in this function
#   - **You are STRICTLY FORBIDDEN** to use other 'run' functions in the definition
#   - **You are STRICTLY FORBIDDEN** to include specific parameter values in 'run' functions
# - **Example:** In 'run_excited_state' definition, you should NOT use 'run_ground_state' directly. Instead, use 'hilbert'/'hamiltonian'/... functions to calculate the ground state.
# - **Note:** You are ALLOWED to define multiple 'run' functions (e.g., run_state_1, run_state_2 for independent calculations)

# ### **2. "item"** - Data Generation/Parameter scanning for Task Items
# - **Purpose:** Generate data/Scan parameters needed in this task
# - **CRITICAL RULES:**
#   - **You are REQUIRED** to call 'run' tagged functions directly. You MUST include at least ONE 'run' function. You are REQUIRED** to ONLY explicitly call 'run' functions to scan parameters
#   - **You are FORBIDDEN** to call explicitly hilbert/hamiltonian/statemodel/compset/observable/effector functions
#   - **You are FORBIDDEN** to call another 'item' function explicitly. 
#   - **You are STRICTLY FORBIDDEN** to include specific parameter values in 'item' functions
# - **Example:** Function that calculates a correlation-interaction dependency in the targeted figure
# - **Note:** You are ALLOWED to define different 'item' functions for different items needed in the task

# ### **3. "plot"** - Data Visualization
# - **Purpose:** Demonstrate given data
# - **You are REQUIRED** to include plot function if your task is to reproduce a figure
# - **Note:** You are ALLOWED to define multiple 'plot' functions


# ====================================================================================================

# ## **Entry point rules**

# ### ** "main"** - Entry Point
# - **Purpose:** Serve as the entry point
# - **CRITICAL RULES:**
#   - **You are FORBIDDEN** to call explicitly hilbert/hamiltonian/observable/statemodel/compset/effector functions
#   - **You are FORBIDDEN** to call 'run' functions
#   - **You are REQUIRED** to call 'item' functions and 'plot' functions DIRECTLY and EXPLICITLY to fulfill the reproduction task
#   - **You are REQUIRED** to include specific parameter values in the 'main' definition
#   - **You are FORBIDDEN** to define GLOBAL parameters
# - **Note:** You MUST choose parameter values according to the target research paper. Choose values according to your own preferences ONLY WHEN the original paper does NOT provide related parameter values.

# **CRITICAL:** You are FORBIDDEN to define functions with other tags.

# ====================================================================================================

# ## **Other Coding Requirements**

# ### **1. Package Usage**
# - **You are REQUIRED** to import netket package (with python command 'import' or 'from ... import'). 
# - **You are REQUIRED** to execute the whole code on CPU, by setting: `os.environ["JAX_PLATFORM_NAME"] = "cpu"`

# ### **2. Plotting**
# - Results from different data_sets and tasks are plotted in a single figure

# ### **3. Character Encoding (CRITICAL)**
# - **You are REQUIRED** to generate code that contains only ASCII characters
# - **You are FORBIDDEN** to use Unicode letters, symbols, or punctuation
# - **Forbidden examples (non-exhaustive):**
#   - Greek letters: ψ, Δ, ω, φ, μ, λ
#   - Math/logic symbols: ÷, ≤, ≥, ≠, ∈, →, ←
#   - Typographic quotes: " " ' '
#   - Primes: ′
#   - Ellipsis: …
#   - Nonbreaking/zero-width spaces
# - **Use ASCII equivalents instead:**
#   - ψ → psi, φ → phi, ω → omega, μ → mu, λ → lambda, Δ → Delta
#   - a ÷ b → div(a, b), ≤ → <=, ≥ → >=, ≠ → !=, ∈ → in
#   - Use straight quotes " ' and plain ASCII punctuation only

# ====================================================================================================

# ## **Output Format**

# You should return a JSON file, which can be transferred to a python dict by json.loads(). The dictionary must contain keys:

# - **"subplot_name"**: the subplot name specified in task plan
# - **"code_type"**: either "python" or "julia" (lowercase string)
# - **"code_content"**: the full source code as a string (plain source, do not include ``` fences)

# **CRITICAL:** You MUST NOT include this prompt itself (including the example below) in your output, in order to avoid the case that the example code itself is extracted as your generated code.

# **Example output** (exact dictionary form, shown here for clarity. You MUST add a "```json" right before the dict and "```" right after the dict! (not shown here) Beside from this dict, NO other information should be output (e.g. this prompt itself should NOT be included in your final response)):
# {{
# "subplot_name": "Fig. 2(a)",
# "code_type": "python",
# "code_content": "import netket as nk\nimport os\nos.environ[\"JAX_PLATFORM_NAME\"] = \"cpu\"\n...(The whole code here)"
# }}

# ====================================================================================================

# ## **Final Checklist**

# 1. **Output format:** Your output MUST BE a complete dictionary (with a '{' and '}'), enclosing by a '```json' and '```'. Even you fail to generate a complete code_content, you must KEEP this output format.

# 2. **Python syntax:** If the code_type is 'python', you MAKE SURE that in the code you use 'def' rather than 'function' to define all functions."""


# url = "https://yinli.one/v1/messages"
# body = {
#     "model": "gemini-3-flash-preview",
#     "messages": [
#         {
#             "role": "user",
#             "content": content
#         }
#     ],
#     "max_tokens": 65536,
#     "temperature": 0.2,
#     "top_p": 0.9,
#     "timeout": 1000
# }


# response = requests.post(
#     url=url,
#     json=body, 
#     headers={
#         "anthropic-version": "2023-06-01",
#         "Authorization": "Bearer sk-9gb0oIfrfLKc5zpBrX5CZjLye7iC3iMYM5FhTqc8lHjFT5h5"
#     }
# )

# print("响应状态码:", response.status_code)  # 打印状态码，方便排查
# print("响应内容:", response.text)


# url = "https://yinli.one/v1/messages"

# body = {
#     "model": "grok-4-fast-reasoning",
#     "messages": [{"role": "user", "content": "你好，测试下通不通"}],
#     "max_tokens": 200,
#     "timeout": 1000,
#     "stream": True
# }

# # headers = {
# #     "Authorization": YIDONG_API_KEY_2,
# #     "Accept": "text/event-stream",
# # }
# headers = {
#         "Content-Type": "application/json", 
#         "anthropic-version": "2023-06-01", 
#         "Authorization": YIDONG_API_KEY_2
#         }


# with requests.post(url, json=body, headers=headers, stream=True, timeout=1000) as r:
#     print("响应状态码:", r.status_code)
#     # 如果希望 4xx/5xx 直接抛异常，可以打开下一行
#     # r.raise_for_status()

#     # 确保按 UTF-8 解码（避免中文乱码）
#     r.encoding = "utf-8"

#     full_text = []
#     event_name = None

#     # 新增：保存总 input / output tokens
#     input_tokens = None
#     output_tokens = None

#     for line in r.iter_lines(decode_unicode=True):
#         if not line:
#             # 空行表示一个 SSE 事件结束
#             continue

#         # line 形如: "event: xxx" 或 "data: {...}"
#         if line.startswith("event:"):
#             event_name = line[len("event:"):].strip()
#             continue

#         if line.startswith("data:"):
#             data_str = line[len("data:"):].strip()

#             # 有些实现会发 "[DONE]"，稳妥处理
#             if data_str == "[DONE]":
#                 break

#             try:
#                 payload = json.loads(data_str)
#             except json.JSONDecodeError:
#                 # data 不是 JSON 就跳过
#                 continue

#             event_type = payload.get("type")

#             # 1) message_start：这里的 usage 在 payload["message"]["usage"] 里
#             if event_type == "message_start":
#                 # print("DEBUG message_start:", payload)  # 如需调试可打开
#                 msg = payload.get("message", {})
#                 u = msg.get("usage", {})         # 可能没有 usage，所以用 get
#                 if "input_tokens" in u:
#                     input_tokens = u["input_tokens"]
#                 # 这里的 u.get("output_tokens") 一般只是初始值，可以忽略

#             # 2) 增量文本 content_block_delta -> delta.text
#             if event_type == "content_block_delta":
#                 delta = payload.get("delta", {})
#                 if delta.get("type") == "text_delta":
#                     chunk = delta.get("text", "")
#                     if chunk:
#                         full_text.append(chunk)
#                         # 想实时打印可以解开下一行
#                         # print(chunk, end="", flush=True)

#             # 3) message_delta / message_stop：通常在这里给最终的 output_tokens
#             if event_type in ("message_delta", "message_stop"):
#                 print("DEBUG message_delta:", payload)
#                 u = payload.get("usage", {})
#                 if "output_tokens" in u:
#                     # 每来一次就覆盖，最后一条就是最终值
#                     output_tokens = u["output_tokens"]

#             # 4) 可选：结束条件（注意要放在读取 usage 之后）
#             if event_type == "message_stop":
#                 break

#     response_text = "".join(full_text)

#     print("\n\n==== 完整结果 ====\n", response_text)
#     print("input_tokens:", input_tokens, "output_tokens:", output_tokens)


# with requests.post(url, json=body, headers=headers, stream=True, timeout=1000) as r:
#     print("响应状态码:", r.status_code)
#     # if r.status_code != 200:
#     #     return None, r.status_code
#     # r.raise_for_status()

#     # 确保按 UTF-8 解码（避免中文乱码）
#     r.encoding = "utf-8"

#     full_text = []
#     event_name = None

#     for line in r.iter_lines(decode_unicode=True):
#         if not line:
#             # 空行表示一个 SSE 事件结束
#             continue

#         # line 形如: "event: xxx" 或 "data: {...}"
#         if line.startswith("event:"):
#             event_name = line[len("event:"):].strip()
#             continue

#         if line.startswith("data:"):
#             data_str = line[len("data:"):].strip()

#             # 有些实现会发 "[DONE]"，稳妥处理
#             if data_str == "[DONE]":
#                 break

#             try:
#                 payload = json.loads(data_str)
#             except json.JSONDecodeError:
#                 # data 不是 JSON 就跳过
#                 continue

#             if payload.get("type") == "message_start":
#                 print("DEBUG message_start:", payload)
#                 usage = payload["message"]["usage"]

#             # 增量文本 content_block_delta -> delta.text
#             if payload.get("type") == "content_block_delta":
#                 delta = payload.get("delta", {})
#                 if delta.get("type") == "text_delta":
#                     chunk = delta.get("text", "")
#                     if chunk:
#                         full_text.append(chunk)
#                         # print(chunk, end="", flush=True)

#             # 可选：结束条件
#             if payload.get("type") == "message_stop":
#                 break

#     response_text = "".join(full_text)
#     print("\n\n==== 完整结果 ====\n", response_text)
#     print("usage:", usage)