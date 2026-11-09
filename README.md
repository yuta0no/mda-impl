# MDA-impl

This repository implements [Model Discovery Agent (MDA)](https://arxiv.org/abs/2608.09696).


## API cost estimation

### TL;DR

For the ChemBench experiment, *conservative estimation* of API cost is <$100 if we use Claude Haiku 5.5 with medium effort.

#### Main experiment with MDA

input: $2,000 (tok / prompt) \times 23 (prompt / seed) \times 3 (seed / world) \times 57 (world) \simeq 8,000,000 (tokens)$

output: $10,000 (tok / prompt) \times 23 (prompt / seed) \times 3 (seed / world) \times 57 (world) \simeq 40,000,000 (tokens)$

total cost: $\simeq \$ 21$

**Note**: LLMs can propose hypotheses in an invalid format. Individually invalid hypotheses are simply discarded (MDA keeps "up to 12 valid, nonduplicate" proposals), so only responses that are unusable as a whole (e.g., unparseable JSON or output truncated at max_tokens) require a new call. If each proposal call is attempted at most N times in total, the worst-case proposal cost is N times the single-attempt estimate.


#### Baseline costs

**ICL-based baselines**: To compute NMSLE, the original paper used only 60 heldout points, rather than the entire held-out 1000 points.
This reduces the cost for ICL-based baselines.

input: $5,000 (\mathrm{tok / prompt}) \times 23 (\mathrm{prompt / seed}) \times 3 (\mathrm{seed / world}) \times 57 (\mathrm{world}) \simeq 20,000,000 (\mathrm{tokens})$

output: $1,000 (\mathrm{tok / prompt}) \times 23 (\mathrm{prompt / seed}) \times 3 (\mathrm{seed / world}) \times 57 (\mathrm{world}) \simeq 4,000,000 (\mathrm{tokens})$

total cost:
- $\simeq \$ 4$ (ICL-MDA)
- $\simeq \$ 8$ (ICL-LLM; additionally needs API to experiment selection)

**LLM-AutoSciLab**: In the ChemBench configuration, each outer iteration makes one LLM call that both updates the working hypothesis (a primary law plus alternatives) and proposes search regions; the active-learning module then selects up to five concrete experiments within those regions. Additional calls propose adversarial validation experiments. Equations are fitted with PySR rather than by the LLM.

In the MDA paper it is evaluated at budgets far below its original setting ($B = 60$) and reaches a 15.8% pass rate vs. MDA's 78.4% at $B = 25$. Given the [PySR](https://github.com/astroautomata/PySR) compute and the availability of the reported numbers, we might as well skip rerunning this baseline.

### Overview

MDA utilises LLMs to expand the model space by prompting an LLM to propose one or more novel models conditioned on the observations and context.
This means we need to pay the API credit to run experiments with MDA.
Since it is important to get a grasp of the estimated API cost before running experiment, we calculate the API cost necessary for the ChemBench experiment with baselines in this section.

Note that while the original paper uses GPT5.6 Luna. In this work, we use Claude Haiku 5.5 as an alternative for some reasons, and Haiku's API cost is $0.10 (input) / $0.50 (output) per 1M tokens for prompts up to 100K tokens ([source](https://platform.claude.com/docs/en/models/haiku-5-5/overview)).

The prompt provided to the LLM can found in [./src/mda_impl/prompts/chem_propose.txt](./src/mda_impl/prompts/chem_propose.txt), and estimated its token count is approximately 400--1500 depending on the history lengths, according to [this unofficial token counter](https://wildandfreetools.com/ai-tools/token-counter/).
There is also an official, free token count API endpoint provided by Anthropic, but it requires you to top-up your API credit first.
(TODO: count API tokens using the official endpoint once I obtain API credits)

For ChemBench, MDA is initialised with three scaler rate observations in $D_0$, which are created by varying $C_A \in \{0.03, 0.3, 3\}$ and holding the other terms constant. It then actively selects $K = 22$ additional experiments.
The hypothesis space is initialised with 9 canonical mechanisms plus up to 12 valid, nonduplicate LLM proposals (= MDA starts with 21 hypothesis at maximum); later triggered calls request six candidates, with a live-pool cap of 24.

### Maximum number of API calls



### Estimated maximum token usage per call

1500 (input) + 