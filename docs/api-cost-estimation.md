# API cost estimation

## TL;DR

For the ChemBench experiment, *conservative estimate* of API cost is <$100 if we use Claude Haiku 5.5 with medium effort.


## Overview

MDA utilises LLMs to expand the model space by prompting an LLM to propose one or more novel models conditioned on the observations and context.
This means we need to pay the API credit to run experiments with MDA.
Since it is important to get a grasp of the estimated API cost before running experiment, we calculate the API cost necessary for the ChemBench experiment with baselines in this section.

Note that while the original paper uses GPT5.6 Luna. In this work, we use Claude Haiku 5.5 as an alternative for some reasons, and Haiku's API cost is $0.10 (input) / $0.50 (output) per 1M tokens for prompts up to 100K tokens ([source](https://platform.claude.com/docs/en/models/haiku-5-5/overview)).

The prompt provided to the LLM can be found in [./src/mda/prompts/chem_propose.txt](../src/mda/prompts/chem_propose.txt). Its size depends on the history length and the pool. Measured with Anthropic's official `count_tokens` endpoint for `claude-haiku-5-5` (`uv run python tools/llm_check.py count`), on the stand-in prompts that the tool rendered from the template before `feat/proposer` added the real renderer (`mda.propose.render_prompt`):

| Prompt | Input tokens | With the structured-output schema (D15) |
| --- | --- | --- |
| Initialisation: 3 observations, no refining section | 504 | 929 |
| Largest expansion: 25 observations, a full pool of 24 forms | 3,512 | 3,937 |

These counts have not been repeated on the real prompts, which the tool now renders. In characters, system prompt included, the stand-in prompts were 953 (initialisation) and 5,566 (largest expansion), and the real ones are 947 and 6,643, so the largest expansion is about 19% longer: each of its 24 forms also names the input its residual correlates with most. The M2 pilot will measure the real prompts' tokens.

So the 2,000 input tokens per prompt assumed below is a middle value, not an upper bound. Even if every call used the largest prompt, input would total $3{,}937 \times 23 \times 3 \times 57 \simeq 15.5$ M tokens, about \$1.55, so output still dominates the cost.

A single real initialisation call on the stand-in prompt (12 hypotheses requested, medium effort, structured output) used 929 input and 2,711 output tokens, thinking included, and cost \$0.0015. That is well under the 10,000 output tokens per prompt assumed below, but it is one sample; the M2 pilot will replace both assumptions with measured averages.

For ChemBench, MDA is initialised with three scaler rate observations in $D_0$, which are created by varying $C_A \in \{0.03, 0.3, 3\}$ and holding the other terms constant. It then actively selects $K = 22$ additional experiments.
The hypothesis space is initialised with 9 canonical mechanisms plus up to 12 valid, nonduplicate LLM proposals (= MDA starts with 21 hypothesis at maximum); later triggered calls request six candidates, with a live-pool cap of 24.


## Main experiment with MDA

input: $2,000 (tok / prompt) \times 23 (prompt / seed) \times 3 (seed / world) \times 57 (world) \simeq 8,000,000 (tokens)$

output: $10,000 (tok / prompt) \times 23 (prompt / seed) \times 3 (seed / world) \times 57 (world) \simeq 40,000,000 (tokens)$

total cost: $\simeq$ $21

**Note**: LLMs can propose hypotheses in an invalid format. Individually invalid hypotheses are simply discarded (MDA keeps "up to 12 valid, nonduplicate" proposals), so only responses that are unusable as a whole (e.g., unparseable JSON or output truncated at max_tokens) require a new call. If each proposal call is attempted at most N times in total, the worst-case proposal cost is N times the single-attempt estimate.


## Baseline costs

**ICL-based baselines**: To compute NMSLE, the original paper used only 60 heldout points, rather than the entire held-out 1000 points.
This reduces the cost for ICL-based baselines.

input: $5,000 (\mathrm{tok / prompt}) \times 23 (\mathrm{prompt / seed}) \times 3 (\mathrm{seed / world}) \times 57 (\mathrm{world}) \simeq 20,000,000 (\mathrm{tokens})$

output: $1,000 (\mathrm{tok / prompt}) \times 23 (\mathrm{prompt / seed}) \times 3 (\mathrm{seed / world}) \times 57 (\mathrm{world}) \simeq 4,000,000 (\mathrm{tokens})$

total cost:

(ICL-MDA) $\simeq$ $4

(ICL-LLM) $\simeq$ $8 (more costly because additionally needs API to experiment selection)

**LLM-AutoSciLab**: In the ChemBench configuration, each outer iteration makes one LLM call that both updates the working hypothesis (a primary law plus alternatives) and proposes search regions; the active-learning module then selects up to five concrete experiments within those regions. Additional calls propose adversarial validation experiments. Equations are fitted with PySR rather than by the LLM.

In the MDA paper it is evaluated at budgets far below its original setting ($B = 60$) and reaches a 15.8% pass rate vs. MDA's 78.4% at $B = 25$. Given the [PySR](https://github.com/astroautomata/PySR) compute and the availability of the reported numbers, we might as well skip rerunning this baseline.
