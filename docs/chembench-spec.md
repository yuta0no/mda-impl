# MDA on ChemBench: implementation spec

This file collects what the MDA paper specifies for the ChemBench experiment, and the defaults we choose where it says nothing. [blueprint.md](./blueprint.md) says which branch implements each part.

Source: [arXiv:2608.09696](https://arxiv.org/abs/2608.09696) **v5**, read from the TeX source. References such as "Alg. 8" or "App. B.2" point to that version. Section, figure and algorithm numbers were recomputed from that source after dropping its commented-out blocks, so they should match the PDF.

## 1. Protocol

Sources: Alg. 1 (agent–environment interface), §4.1, App. B.1 and B.2.

| Item | Value | Source |
| --- | --- | --- |
| Worlds | 57 (`mda.bench.list_worlds()`): 9 canonical, 48 compound | §4.1 |
| Difficulty, noise | medium, 1% multiplicative | App. B.1 |
| Law version | `v0`. The paper does not say which; `mda.bench` pins it (see README) | — |
| Replicates | 3 seeds per world, so 171 (world, seed) cells. The seed changes fitting, design search and the held-out query set, not the noise draw | App. B.1 |
| Passive data $D_0$ | 3 points: $C_A \in \{0.03, 0.3, 3\}$ with $(C_I, C_B, C_P, \mathrm{Enz}, T, \mathrm{pH}) = (0, 1, 0, 1, 310, 7)$ | App. B.2 |
| Active experiments | $K = 22$, batch size 1 | §4.1 |
| Budget $B$ | total number of observations, $B = 3 + k$, so $B \in \{3, \dots, 25\}$ | Fig. 2 |
| Evaluation | after every update, giving a learning curve | App. A.1 |

One cell runs:

```text
agent = MDA.init(llm, context, D0)
evaluate(agent.get_predictor())          # B = 3
for k in 1..22:
    x = agent.design()
    y = world.observe(x)
    agent.update_state(x, y)
    evaluate(agent.get_predictor())      # B = 3 + k
```

## 2. Hypotheses and the seed library

A hypothesis is an algebraic rate law $r_0 = f(C_A, C_I, C_B, C_P, \mathrm{Enz}, T, \mathrm{pH}; \theta)$ whose parameters have names and bounds. This is the JSON format of the proposer prompt, and `mda.hypothesis` already parses and compiles it. The parameter prior is uniform over the declared bounds (App. A.4).

The seed library is "nine canonical executable laws" (App. B.2). We take these to be the forms of the nine canonical benchmark worlds, which `mda.bench.canonical_worlds()` returns (the benchmark excludes `c4_ph_activity`). Parameter names are those of the vendored oracle, so fitted values compare with a world's constants by name:

| Seed | `expr` |
| --- | --- |
| Michaelis–Menten | `kcat*Enz*C_A/(Km + C_A)` |
| Competitive inhibition | `kcat*Enz*C_A/(Km*(1 + C_I/Ki) + C_A)` |
| Product inhibition | `kcat*Enz*C_A/(Km*(1 + C_P/Kp) + C_A)` |
| Arrhenius | `kcat_ref*exp(-Ea/8.314*(1/T - 1/310))*Enz*C_A/(Km + C_A)` |
| Ping-pong bisubstrate | `kcat*Enz*C_A*C_B/(KmA*C_B + KmB*C_A + C_A*C_B)` |
| Uncompetitive inhibition | `kcat*Enz*C_A/(Km + C_A*(1 + C_I/Ki))` |
| Substrate inhibition | `kcat*Enz*C_A/(Km + C_A + C_A**2/Ki)` |
| Hill | `kcat*Enz*C_A**n/(K_half**n + C_A**n)` |
| Noncompetitive inhibition | `kcat*Enz*C_A/((1 + C_I/Ki)*(Km + C_A))` |

The paper gives no parameter bounds for the seeds; see [D7](#10-where-the-paper-is-silent).

## 3. Fitting and model weights

The primary arm uses point estimates and BIC weights. Sources: App. A.6.3 (BIC approximation, Alg. 8), App. A.8 (Alg. 9), App. B.2.

- **Point estimate.** $\hat\theta_m = \arg\min_\theta \mathrm{RSS}_m(\theta)$ over the declared bounds, by bounded multistart nonlinear least squares. Every model is refitted on all data at every update.
- **Whitened residuals.** $\mathrm{RSS}_m(\theta) = \sum_n \big((y_n - \hat y_n(\theta)) / \sigma_n\big)^2$ with $\sigma_n = \max(\rho |y_n|, \sigma_{\mathrm{floor}})$, $\rho = 0.03$, $\sigma_{\mathrm{floor}} = 0.01$.
- **Profiled-scale BIC** (the headline configuration): $\mathrm{BIC}_m = N \log(\mathrm{RSS}_m / N) + C_m \log N$, where $C_m$ is the number of free parameters ([D20](#10-where-the-paper-is-silent)).
- **Fixed-scale BIC** ($\tau = 1$, a control): $\mathrm{BIC}_m = \mathrm{RSS}_m + C_m \log N$.
- **Weights.** $p(m \mid D) = \mathrm{softmax}_m(-\mathrm{BIC}_m / 2 + \log p(m))$ with $p(m) \propto e^{-\lambda C_m}$ and $\lambda = 0$ for the BIC arm (Table 2).
- **Pruning.** Keep the $N_m$ models with the largest weights and renormalise. A model whose RSS is not finite has weight 0 and is pruned first.
- **Prediction.** Take the current-data MAP structure $\hat m = \arg\max_m p(m \mid D)$ and predict $f_{\hat m}(x; \hat\theta_{\hat m})$ (§4.1; `get-model` and `get-predictor` in Alg. 2). There is no model averaging.

## 4. Pool management

- **Initialisation (Alg. 11).** The pool starts as the 9 seeds plus one LLM call requesting $N_{\mathrm{init}} = 12$ laws, using the prompt without its refining section. Proposals that fail validation are dropped: they do not compile or their bounds are invalid. Exact-expression duplicates are removed, so the initial pool has at most 21 models. If the pool ends up empty, retry within a limit or report failure.
- **Expansion (Alg. 12).** One LLM call requests $N_{\mathrm{new}} = 6$ "residual-directed" proposals. Its prompt adds the refining section: the forms already in the pool with their median relative residuals, and the inputs the best form's residual correlates with. It also adds the remaining budget and phase. The new candidates are fitted, and the next model-posterior call rescores and prunes the whole pool. After initialisation, this is the only LLM call.
- **Adaptive cap (Alg. 10; App. B.2, "Adaptive pool size and pruning").** After each update, the cap drops to $N_{\min} = 6$ when two conditions hold: the largest weight exceeds $\tau_c = 0.9$, and the MAP model's median relative fit residual is at most $\tau_{\mathrm{fit}} = 0.05$. The 6 models with the largest weights are kept and the pool is refitted. Otherwise the cap is $N_{\max} = 24$. Raising the cap again does not restore discarded models.

## 5. Update step

Sources: Alg. 2 (`update-state`) and Alg. 13 (`predictive-check`).

```text
update_state(x, y):
    e = predictive_check(x, y)                # which fit it uses: see D5
    D = D ∪ {(x, y)}
    refit every model, recompute weights, prune to the cap
    for i in 1..R_m:
        if e <= tau_e or n_e >= N_e_max: break
        pool += llm.expand(pool, D, context, N_new=6)
        refit, reweight, prune
        n_e += 1
        e = in-sample check of the refitted MAP model
    adapt_and_prune()
```

The per-datum relative error is $r_j = |F(y_j) - \hat F(x_j)| / \max(|F(y_j)|, c)$. Then $e^*$ is the median of $r_j$ over the new datum and, with back-testing (on by default), over the stored data. Expansion happens when $e^* > \tau_e$.

In Alg. 13, the new datum is scored prequentially by the fit made before it arrived. The paper notes that the executed ChemBench agent did something else (App. A.8). It gated its first expansion on residuals cached from the preceding fit, then refitted on all data and rechecked if it expanded again. So a surprising observation can trigger expansion one update later.

## 6. Experiment design

Sources: App. B.2 (the objective $V(a)$) and App. A.9 (Alg. 14).

$$V(a) = \sum_m p(m \mid D) \Big(\bar\mu_m(a) - \sum_{m'} p(m' \mid D)\, \bar\mu_{m'}(a)\Big)^2$$

In the BIC arm, $\bar\mu_m(a) = f_m(a; \hat\theta_m)$. $V(a)$ is the model-disagreement surrogate for expected information gain, without the noise whitening.

- **Optimiser.** CMA-ES with 48 objective evaluations per round on the normalised 7-d box. The paper says the "concentration and enzyme axes are mapped logarithmically and temperature and pH linearly". `mda.bench.to_unit` and `from_unit` follow the benchmark instead: $C_A$, $C_B$ and $\mathrm{Enz}$ are logarithmic, and $C_I$ and $C_P$ are linear because their lower bound is 0. Batch size is 1, and the pool is refitted before the next design.
- **Random-design control.** Draw uniformly in the unit box, the same distribution as `World.heldout`.

## 7. Evaluation and reporting

Sources: App. B.1, App. B.4, Fig. 2 caption.

- **RMSLE and pass.** `mda.metrics.rmsle` is computed on the benchmark's fixed test grid, `World.heldout(42, 1000)`, the set `evaluate_law` uses. A cell passes when RMSLE < 0.05. A law that cannot be evaluated over the test range counts as a failure.
- **nMSLE.** `mda.metrics.nmsle` is computed on a held-out query set per (world, seed). The visible text only says that the seed changes the query conditions. A commented-out paragraph of the source gives its size as 60.
- **Aggregation.** For the main curve (Fig. 2a), average over seeds within a world arithmetically, then take the geometric mean over worlds. 95% intervals come from 5,000 world-bootstrap replicates. Pass rates use all 171 cells, with failures counted. The paper reports paired ratios between methods with world-bootstrap intervals, but does not give the estimator; see D19. The controls figure (Fig. 7) aggregates differently: geometric means over worlds and replicates.
- **Cohort.** The paper's nMSLE curve uses 115 cells in 54 worlds: those where the corrected LLM-AutoSciLab is evaluable at every budget. We do not rerun LLM-AutoSciLab, so we cannot rebuild that cohort. We report nMSLE on the cells where every compared method is finite, and say so.

## 8. Arms and the paper's numbers

| Arm | What differs from the primary arm | Paper result |
| --- | --- | --- |
| MDA hybrid (primary) | — | Pass rate 74.9% at $B = 20$ and 78.4% at $B = 25$, over all 171 cells |
| Fixed library | 9 seeds only, no LLM calls, no expansion | At $B = 15$, hybrid minus fixed pass rate is +33.3 points (95% CI +22.8 to +43.9), and +39.6 points on the 48 out-of-library worlds. The hybrid/fixed nMSE ratio is 0.059. Both arms pass every cell of the 9 canonical worlds |
| Random design | Designs drawn uniformly | At 15 observations, the CMA-ES/random nMSE ratio is 0.314 under BIC and 0.132 under SMC |
| Fixed-scale BIC | $\tau = 1$ | Better than profiled-scale BIC at small budgets, similar at 15 observations |
| SMC control | Tempered SMC evidence with 100 particles per structure; $p(m) \propto e^{-2C_m}$; $\bar\mu_m$ averaged over particles | BIC+CMA vs SMC+CMA nMSE ratio is 0.334 |
| ICL (+MDA design) | The LLM predicts the query outcomes from a prefix of MDA's history | Worse than MDA, and slightly worse than ICL (+LLM design) |
| ICL (+LLM design) | Starts from $D_0$; the LLM picks all 22 experiments, then predicts | Worse than MDA |
| LLM-AutoSciLab | LLM + PySR, rerun at each $B \in \{5, 10, 15, 20, 25\}$ | 15.8% pass at $B = 20$ and 25. Not rerun here; see [api-cost-estimation.md](./api-cost-estimation.md) |

The controls in Fig. 7 (random design, fixed-scale BIC, SMC) use 12 predeclared worlds × 3 seeds, with 3 + 12 observations. The paper does not name the 12 worlds.

Table 1 reports MDA's RMSLE at $B = 15$ on three worlds: 0.002 (Michaelis–Menten), 0.033 (MM + competitive inhibition + Arrhenius) and 0.115 (fractal kinetics, a fail).

## 9. LLM

- **Paper.** GPT-5.6-Luna at medium reasoning effort. Temperature and the number of samples are not stated.
- **Ours.** Claude Haiku 5.5 (`claude-haiku-5-5`) at medium effort, as decided in [api-cost-estimation.md](./api-cost-estimation.md). Haiku 5.5 rejects non-default sampling parameters, so we set no temperature, and the effort is set explicitly.
- **Prompt.** [`src/mda/prompts/chem_propose.txt`](../src/mda/prompts/chem_propose.txt) is App. G.1, verbatim. The paper has no ChemBench prompt for the ICL baselines.

## 10. Where the paper is silent

Each default below is ours. In code, put it in `MDAConfig` with a comment naming its ID.

| ID | Open question | Our default | Alternative to keep behind a flag |
| --- | --- | --- | --- |
| D1 | Which residual is fitted? App. A.6.3 whitens raw residuals, but App. B.2 says both arms "fit residuals in log-rate space" | The whitened raw residual $(y - \hat y)/\sigma_n$. With $\rho = 0.03$, it approximates a scaled log-space residual once $\lvert y\rvert \gg 0.3$ | `log1p` residuals |
| D2 | Profiled BIC is $-\infty$ when $\mathrm{RSS} = 0$, which happens at $B = 3$ whenever $C_m \ge 3$ | Floor $\mathrm{RSS}/N$ at $10^{-12}$, so the exact fits tie and $C_m \log N$ decides between them | — |
| D3 | Expansion threshold $\tau_e$ | 0.05, the scale used by $\tau_{\mathrm{fit}}$ and the pass threshold | — |
| D4 | $R_m$ and $N_e^{\max}$ | $R_m = 1$ and $N_e^{\max} = \infty$: at most 22 expansions per cell, the 23 calls per cell that [api-cost-estimation.md](./api-cost-estimation.md) assumes | — |
| D5 | Which predictive-check variant, which target $F$, and which floor $c$? | The variant the executed ChemBench agent used: cached residuals from the preceding fit. Relative error on the raw rate ($F(y) = y$) with $c = 0.01$ | The prequential variant of Alg. 13 |
| D6 | Prompt fields the template leaves open | Residual list: each pool form with its in-sample median relative residual. Correlation hint: the one or two inputs with the largest absolute Spearman correlation between the MAP model's signed residual and the input in unit coordinates. Budget: $K - k$. Phase: `refine` if the MAP model's median relative residual is at most $\tau_{\mathrm{fit}}$, otherwise `explore` | Omit the phase line, which the template marks as optional |
| D7 | Seed-law parameter bounds | Set from physical scales and the input box, not from the worlds' constants: `kcat` and `kcat_ref` $[10^{-3}, 10^{3}]$; `Km`, `Ki`, `Kp`, `KmA`, `KmB`, `K_half` $[10^{-3}, 10^{3}]$ mM; `n` $[0.1, 8]$; `Ea` $[10^{3}, 2 \times 10^{5}]$ J/mol. A test then checks, without tuning, that the nine canonical worlds' constants fall inside | — |
| D8 | Multistart settings | `scipy.optimize.least_squares` (`method="trf"`) with bounds. Parameters whose bounds span more than two decades are optimised in log space. 8 starts drawn from the bounds, plus a warm start from the previous step's $\hat\theta$. The RNG is seeded by (cell seed, step) | — |
| D9 | In which space is $V(a)$ computed? | Raw rate, as written in App. B.2 | $\log(1 + y)$, which weights low-rate regions more |
| D10 | CMA-ES settings | pycma with mean at the box centre, $\sigma_0 = 0.3$, default population size (9 in 7-d) and bounds $[0, 1]^7$, stopped after exactly 48 evaluations. Return the best point evaluated, not the final mean | — |
| D11 | Query-set seeding | 60 conditions from `World.heldout`, seeded from (world, seed) with `zlib.crc32` (not the salted `hash`). All arms share them, and the seed is independent of the fitting and design RNG | — |
| D12 | The 12 control worlds are not named | Run the controls on all 57 worlds if compute allows. Otherwise, fix a list of 12 in a config file before running | — |
| D13 | What counts as a duplicate | Exact expression after normalising formatting only: strip comments and whitespace, then `ast.unparse(ast.parse(expr))`. Renaming a parameter makes a new expression, as in the paper | — |
| D14 | Unusable LLM replies (`ProposalParseError`, `stop_reason` of `max_tokens` or `refusal`) | Up to 3 attempts per call, all logged. After that, continue without new candidates; at initialisation, the 9 seeds | — |
| D15 | Output format | Structured outputs (`output_config.format`) with the hypotheses JSON schema, leaving the prompt text unchanged. `parse_hypotheses` still validates every expression | The plain prompt, as in the paper |
| D16 | ICL prompts | ICL (+MDA design): the paper's generic ICL forecaster template (App. G.2, BoxingGym) with the ChemBench world description from `chem_propose.txt`, predicting the 60 queries. ICL (+LLM design): ask for one condition inside the input bounds. After one format-only retry, a malformed reply becomes a random condition. A missing or unparseable forecast counts as a failure | — |
| D17 | Reproducibility of LLM calls | Cache each reply keyed by (arm, world, seed, call index, prompt hash), so a rerun replays the reply instead of querying again | — |
| D18 | SMC settings not given: target ESS fraction $\eta$, moves per rung $R_p$, rung cap $J_p$ | $\eta = 0.5$, $R_p = 5$, $J_p = 100$. Proposals outside the bounds are rejected. The proposal scale is half the per-coordinate particle standard deviation, floored at $10^{-3}$ before scaling (Alg. 6 caption) | — |
| D19 | Estimator for paired ratios between two methods | Average the paired log-ratios of the seeds within each world, take the mean over worlds, exponentiate, and bootstrap worlds | Ratio of the two methods' aggregated curves |
| D20 | Does a parameter whose bounds coincide count towards $C_m$? | No. $C_m$ counts the free parameters, those with `low < high`. The fit holds a parameter with `low == high` fixed, so it adds no flexibility to penalise | — |

## 11. Inconsistencies and pitfalls in the paper

- **Pass threshold.** The AutoSciLab paper's text says 0.01, but its code and the MDA paper use 0.05 (App. B.1).
- **Empty initial batch.** App. B.2 says that in the Fig. 2b storyboard, five new expressions take the pool "from nine to 14" at $B = 6$. A pool of 9 after initialisation suggests that none of the 12 initial proposals survived in that run. This is our reading, not something the paper states, but the code must handle an empty initial batch either way.
- **SMC bounds.** The paper's NumPy SMC fitter clipped proposals at the bounds. That violates detailed balance (Alg. 6 caption), so our implementation must reject out-of-bounds proposals instead.
- **Data-dependent bounds.** The proposer chooses bounds after seeing the data, which inflates evidence (App. A.4 footnote). The weights are approximate predictive weights, not calibrated posterior probabilities.
- **Near-duplicates.** Exact-expression matching does not merge equivalent laws, so weight can split between near-duplicates (App. B.2).
- **Baseline bug.** The upstream LLM-AutoSciLab selector compared training scores against validation scores. The paper fixed this for its baseline runs (App. B.3).
- **Benchmark defects.** The upstream ChemBench defects listed in the [README](../README.md#upstream-code-defects) apply to every arm.
