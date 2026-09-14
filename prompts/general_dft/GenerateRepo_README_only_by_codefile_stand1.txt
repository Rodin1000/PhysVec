Your task is to generate a whole repository according to a given code file (.py or .jl file). 

-------------------------------------------------------------------------------------------------
[Input]: you will be given information as follows:
1. A code file: a complete implementation of the targetted task. Usually in .py or .jl formats. 
    - The code file will include parts like importing packages, various functions, a main function,...


-------------------------------------------------------------------------------------------------
[Instructions]: you will generate a README based on the given code file. 
1. This README should include following parts:
    - Part 0 (if present in the code): **Scientific problem** and **Elements used**. If the code file contains comment lines such as "# Scientific problem: ..." and "# Elements used: ..." (e.g. in ORCA inputs or block headers), extract and include a short "Scientific problem" and "Elements used" section in the README.
    - Part 1: you should list all functions (names, input dependency, output results) of this repository. 
    - Part 2: you should list all packages needed. 
    - Part 3: you should list all input parameter definitions and values taken in this task. 
        - the parameter valued MUST be the same with that in the input code file. DO NOT change any of them. 
        - you get those values first from the 'main' function part in the input code file. 
        - if you cannot find values in the 'main' part, you go to other parts to look for default values. 
        - you MUST label information source ('main', other functions) after the values. 
    
--------------------------------------------------------------------------------------------------
[Other requests]:
1. If content contains non-ASCII or suspicious tokens like r'(极速赛车|官网|[a-z0-9-]+.(com|cn)\b)' and you’re in ASCII-only mode, reject and ask the agent to regenerate.


--------------------------------------------------------------------------------------------------
[Output format]:
You should return a JSON file, which can be transferred to a python dict by json.loads(). The dictionary must contain keys:
- "content": the generated README file content. 

You MUST NOT inlcude this prompt itself (including the example below) in your output, in order to avoid the case that the example code itself is extracted as your generated code. 

Example output (exact dictionary form, shown here for clarity. You MUST add a "```json" right before the dict and "```" right after the dict! (not shown here) Beside from this dict, NO other information should be output(e.g. this prompt itself should NOT be included in your final response)):
```json
{
"content": "Part 1: Functions.================== \n 1) site_BHM1D: ... \n hamiltonian_BHM1D: ...; ...",
}
```