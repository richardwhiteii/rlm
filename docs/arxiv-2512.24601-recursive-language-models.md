# Recursive Language Models

**arXiv:2512.24601** | Submitted: December 31, 2025

## Authors
- Alex L. Zhang
- Tim Kraska
- Omar Khattab

## Abstract

We study allowing large language models (LLMs) to process arbitrarily long prompts through the lens of inference-time scaling. We propose Recursive Language Models (RLMs), a general inference strategy that treats long prompts as part of an external environment and allows the LLM to programmatically examine, decompose, and recursively call itself over snippets of the prompt.

The research demonstrates that RLMs successfully handle inputs extending far beyond standard model context windows and substantially improve performance on diverse long-context tasks while maintaining comparable computational efficiency.

## Key Findings

- Processes inputs up to **100 times longer** than model context windows
- Outperforms base LLMs and common long-context solutions across four task categories
- Maintains comparable or lower cost per query

## Metadata

- **Classification**: Artificial Intelligence (cs.AI); Computation and Language (cs.CL)
- **Format**: 9 pages (33 with appendix)
- **License**: CC BY 4.0
- **DOI**: https://doi.org/10.48550/arXiv.2512.24601
- **HTML**: https://arxiv.org/html/2512.24601v1
- **PDF**: https://arxiv.org/pdf/2512.24601

---

## 1. Introduction

### Context Limitations and Core Problem

Modern LLMs face two persistent challenges: limited context lengths and context rot—performance degradation as context grows. While training improvements may gradually increase context windows, researchers ask whether dramatic scaling is achievable through inference-time computation rather than architectural changes.

### Key Innovation: Environment-Based Processing

RLMs adopt an "out-of-core algorithm" approach inspired by data systems managing memory constraints. The fundamental insight: "long prompts should not be fed into the neural network directly but should instead be treated as part of the environment that the LLM can symbolically interact with."

### How RLMs Work

An RLM exposes the standard LLM interface (accepting a prompt, returning a response) but internally:
- Initializes a Python REPL environment
- Stores the input prompt as a variable in that environment
- Provides the LLM general context (prompt length, structure)
- Encourages the LLM to write code that inspects, decomposes, and recursively calls itself

This design circumvents prior limitations of recursive decomposition approaches, which couldn't scale inputs beyond the underlying model's context window.

---

## 2. Methodology: Scaling Long Context Tasks

### Task Framework

The research characterizes long-context tasks by information density and complexity scaling:
- **Constant complexity**: Single Needle-in-Haystack (S-NIAH)
- **Linear complexity**: OOLONG (requires examining nearly all entries)
- **Quadratic complexity**: OOLONG-Pairs (requires processing pairs of entries)

This framing explains why frontier models degrade faster on more complex tasks at identical input lengths.

### Evaluated Tasks

1. **S-NIAH**: Finding specific phrases in large unrelated text (50 tasks)
2. **BrowseComp-Plus**: Multi-hop QA requiring synthesis across 1000 documents (150 tasks, 6-11M tokens)
3. **OOLONG**: Semantic transformation requiring line-by-line analysis (50 tasks, 131K tokens)
4. **OOLONG-Pairs**: Modified OOLONG requiring pairwise aggregation (20 tasks, 32K tokens)
5. **CodeQA**: Code repository understanding from LongBench-v2

### REPL Environment Design

The Python REPL provides:
- A `context` variable containing the full input prompt as a string
- An `llm_query()` function enabling recursive LLM calls
- Standard `print()` for inspection
- Full programmatic access (slicing, regex, chunking)

The LLM receives information about total context length and chunk structure, enabling intelligent decomposition strategies.

### Baselines and Methods

- **Base Models**: Direct LLM calls (GPT-5, Qwen3-Coder-480B)
- **Summary Agent**: Iterative context summarization, chunking when input exceeds window
- **CodeAct + BM25**: Code-executing agent with retrieval-based context management
- **RLM with REPL**: Full proposed method with recursive sub-calls
- **RLM without sub-calls**: Ablation to isolate REPL contributions

---

## 3. Emergent Behavior Patterns in RLM Trajectories

### Filtering Without Full Inspection

Models develop strategies using priors to narrow search spaces. Example: "RLM(GPT-5) using regex queries search for chunks containing keywords...then filtering based on observations."

### Recursive Chunking Strategies

Rather than uniform partitioning, models chunk by natural boundaries (newlines, document breaks) and recursively query sub-LMs on each chunk, storing results in variables for aggregation.

### Answer Verification Loops

Some models implicitly avoid context rot by using fresh sub-LM calls for verification, though this sometimes creates redundant, cost-inflating iterations.

### Long Output Handling

RLMs generate outputs exceeding model limits by returning variables constructed in the REPL environment as mixtures of programmatic and sub-LLM outputs.

---

## 4. Experimental Results

### Key Findings

**Observation 1**: RLMs scale to 10M+ token regimes while maintaining comparable or cheaper costs than summarization baselines—"up to 3× cheaper while maintaining stronger performance."

**Observation 2**: The REPL environment proves necessary for scaling beyond context limits. Ablation results show recursive sub-calling provides substantial benefits primarily on information-dense tasks (10-59% improvements on OOLONG/OOLONG-Pairs).

**Observation 3**: Performance degradation correlates with task complexity. GPT-5 degrades significantly faster on OOLONG and OOLONG-Pairs than S-NIAH, while RLM performance maintains relatively shallow degradation curves.

**Observation 4**: Cost variance remains high due to trajectory-length diversity. Median RLM costs are competitive with base models, but 95th-percentile costs spike significantly.

**Observation 5**: Different models exhibit distinct context-management strategies. GPT-5 conservatively uses sub-calls (~10 per task), while Qwen3-Coder sometimes generates thousands, requiring explicit prompting constraints.

### Performance Table Summary

| Benchmark | Model | Base | CodeAct | Summary | RLM |
|-----------|-------|------|---------|---------|-----|
| CodeQA | Qwen3 | 20% | 24% | 50% | **56%** |
| BrowseComp+ | Qwen3 | 0% | 12.66% | 38% | **44.66%** |
| OOLONG | Qwen3 | 36% | 38% | 44.06% | **48%** |
| OOLONG-Pairs | Qwen3 | 0.06% | 0.28% | 0.31% | **23.11%** |
| CodeQA | GPT-5 | 24% | 22% | 58% | **62%** |
| BrowseComp+ | GPT-5 | 0% | 51% | 70.47% | **91.33%** |
| OOLONG | GPT-5 | 44% | 38% | 46% | **56.50%** |
| OOLONG-Pairs | GPT-5 | 0.04% | 24.67% | 0.01% | **58%** |

Results show particularly dramatic improvements on OOLONG-Pairs—base models achieve <0.1% while RLMs achieve 23-58%.

---

## 5. Related Work

RLMs advance beyond two prior directions:
- **Architectural approaches**: Retraining models with longer context windows
- **Scaffold approaches**: Context condensation, memory hierarchies, or task decomposition

Unlike prior recursive decomposition work (THREAD, ReDel, Context Folding), "RLMs are enabled by an extremely simple intuition...to symbolically manipulate arbitrarily long strings and to iteratively refine their recursion via execution feedback."

---

## 6. Limitations and Future Work

### Current Constraints

- All sub-calls implemented synchronously, causing runtime inefficiency
- Recursion depth limited to one layer (sub-LMs don't spawn further RLMs)
- Final answer detection via brittle tag-wrapping (FINAL/FINAL_VAR)
- Model-specific system prompts required (Qwen3 needed explicit sub-call warnings)

### Future Directions

- Asynchronous sub-call execution for runtime improvement
- Deeper recursion trees
- Explicit training of RLM-optimized models
- Viewing RLM trajectories as trainable reasoning patterns

---

## 7. Conclusion

RLMs provide "a general inference framework for language models that offloads the input context and enables language models to recursively sub-query language models before providing an output." The approach achieves "dramatic scaling of context size...by orders of magnitude," demonstrating that inference-time computation can overcome architectural limitations when combined with symbolic environment interaction.

---

## Key Technical Innovations

1. **Environment offloading**: Context as REPL variable, not tokens
2. **Symbolic manipulation**: Code-based decomposition with programmatic control
3. **Recursive sub-calling**: LLMs deciding when/how to defer to sub-LMs
4. **Mixed output generation**: Variables combining code and LLM outputs

---

## Applicability to This Project

This MCP server implements the RLM pattern for Claude Code:

| Paper Concept | Implementation |
|---------------|----------------|
| REPL environment | `rlm_load_context` + `rlm_exec` |
| `context` variable | Stored contexts in memory/disk |
| `llm_query()` function | `rlm_sub_query` / `rlm_sub_query_batch` |
| Chunking strategies | `rlm_chunk_context` (lines/chars/paragraphs) |
| Result aggregation | `rlm_store_result` / `rlm_get_results` |
| Filtering | `rlm_filter_context` (regex) |

The server enables Claude to autonomously apply RLM patterns when facing large contexts, achieving the paper's goals within the Claude Code ecosystem.
