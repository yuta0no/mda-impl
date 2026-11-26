# Implementation blueprint

This blueprint lists what remains to reproduce the Model Discovery Agent (MDA) experiment on ChemBench ([arXiv:2608.09696v5](https://arxiv.org/abs/2608.09696), §4.1 and App. B), and which branch implements each part. [chembench-spec.md](./chembench-spec.md) gives the paper's settings and our defaults where the paper is silent (D1–D20); this file refers to it as "spec §n".

**In scope:**

- the primary MDA arm (hybrid seed library + LLM, BIC weights, CMA-ES design);
- the fixed-library, random-design, fixed-scale-BIC and SMC controls;
- both ICL baselines;
- pass rates and nMSLE curves that can be compared with the paper.

**Out of scope:**

- GlucoseBench, NeuronBench and BoxingGym;
- particle filters and synthetic likelihoods, which only latent dynamics need;
- rerunning LLM-AutoSciLab ([api-cost-estimation.md](./api-cost-estimation.md) explains why).

## Current state

This is `main` with `feat/fit`, `feat/library`, `feat/posterior` and `feat/design` merged.

| Module | Contents | Merged in |
| --- | --- | --- |
| `mda.bench` | `World` (`observe`, `truth`, `heldout`), `to_unit` and `from_unit`, benchmark constants | `feat/chembench` (#1) |
| `mda.metrics` | `rmsle`, `nmsle`, `passed` | `feat/chembench` (#1) |
| `mda.arrays` | jaxtyping aliases | `feat/chembench` (#1), `feat/posterior` |
| `mda.hypothesis` | `Hypothesis`, `Param`, `Model`, `compile_hypothesis`, `normalise_expr`, `parse_hypotheses` | `feat/hypothesis` (#4), `feat/fit` (#8), `feat/posterior` |
| `mda.fit` | `fit` and `FitResult`, `sigma`, `residuals`, `relative_residuals`, `residual_input_spearman` | `feat/fit` (#8) |
| `mda.library` | `SEED_LIBRARY`, the 9 seed laws with the D7 bounds, and `SEED_BY_WORLD`, mapping each canonical world to its seed | `feat/library` (#9) |
| `mda.pool` | `Candidate`, `Origin` and `Pool`: `add` with exact-expression dedupe, `refit`, `keep`, and the per-model arrays `rss`, `n_free` and `fit_err` | `feat/posterior` |
| `mda.posterior` | `bic_profiled`, `bic_fixed`, `log_prior`, `log_weights`, `weights`, `prune_top`, `adaptive_cap` | `feat/posterior` |
| `mda.design` | `between_model_variance`, `cma_design`, `random_design` | `feat/design` |
| `prompts/chem_propose.txt` | The proposer prompt, verbatim from App. G.1 | — |
| CI | pre-commit, ty, and pytest on Python 3.12 and 3.13 | `dev/ci` (#2), `dev/ty` (#3) |
| Tooling | ty as a pre-commit hook, and the Development section of the README | `dev/ty-precommit` (#5) |

## Target layout

```text
src/mda/
  bench.py, metrics.py, arrays.py, hypothesis.py, fit.py, library.py   done
  pool.py, posterior.py, design.py                                     done
  config.py     feat/agent        MDAConfig: every hyperparameter, each citing the paper or a D-number
  agent.py      feat/agent        MDAAgent: init, design, update_state, get_model, get_predictor (Alg. 2)
  propose.py    feat/agent        Proposer protocol and NullProposer
                feat/proposer     LLMProposer: prompt rendering, parse, dedupe, retry
  llm.py        feat/llm-client   Claude client wrapper, reply cache, transcripts, usage and cost
  run.py        feat/agent        run_cell: Alg. 1 for one (world, seed), with a JSON run log
                feat/runner       grid over worlds × seeds × arms, CLI, resume, manifest
  smc.py        feat/smc          tempered-SMC evidence (control arm)
  icl.py        feat/icl          ICL (+MDA design) and ICL (+LLM design)
  analysis.py   feat/analysis     aggregation, world bootstrap, figures and tables
configs/        frozen arm configs (feat/runner, exp/chembench)
runs/, cache/   generated run logs and LLM replies; git-ignored
```

One `update_state` call, in the primary arm:

```mermaid
flowchart LR
    obs["observe y at x"] --> chk["predictive check e*"]
    chk --> add["D ← D ∪ {(x, y)}"]
    add --> fit["refit pool, BIC weights, prune to cap"]
    fit --> gate{"e* > τ_e ?"}
    gate -- yes --> llm["LLM proposes 6 laws"] --> fit2["refit, reweight, prune"] --> adapt
    gate -- no --> adapt["adaptive cap: 6 or 24"]
    adapt --> design["CMA-ES maximises V(a) → next x"]
```

## Branch plan

```mermaid
flowchart TD
    main(["main"]) --> fit["feat/fit (done)"]
    fit --> library["feat/library (done)"]
    fit --> posterior["feat/posterior (done)"]
    main --> design["feat/design (done)"]
    main --> llmc["feat/llm-client"]
    library --> agent["feat/agent"]
    posterior --> agent
    design --> agent
    agent --> runner["feat/runner"]
    agent --> proposer["feat/proposer"]
    llmc --> proposer
    runner --> analysis["feat/analysis"]
    runner --> icl["feat/icl"]
    llmc --> icl
    posterior --> smc["feat/smc"]
    design --> smc
    proposer --> exp["exp/chembench"]
    analysis --> exp
    icl --> exp
    smc --> exp
```

There are three tracks, which can proceed in parallel:

- **Numerics:** `feat/fit`, `feat/library`, `feat/posterior` and `feat/design` (done) → `feat/agent` → `feat/runner`.
- **LLM:** `feat/llm-client` can start now. `feat/proposer` waits for `feat/agent`.
- **Evaluation:** `feat/analysis`, `feat/icl` and `feat/smc` start once the run-log schema from `feat/agent` is fixed.

| # | Branch | Scope | Depends on | Milestone |
| --- | --- | --- | --- | --- |
| 1 | `feat/fit` (done) | Bounded multistart least squares and residual diagnostics | main | M1 |
| 2 | `feat/library` (done) | The 9 canonical seed laws | `feat/fit` | M1 |
| 3 | `feat/posterior` (done) | Pool, BIC, weights, pruning, adaptive cap | `feat/fit` | M1 |
| 4 | `feat/design` (done) | $V(a)$, CMA-ES, random design | main | M1 |
| 5 | `feat/agent` | MDA agent loop, predictive check, one-cell runner | 2, 3, 4 | M1 |
| 6 | `feat/runner` | Grid runner, CLI, arm configs | 5 | M1 |
| 7 | `feat/llm-client` | Claude wrapper, reply cache, cost accounting | main | M2 |
| 8 | `feat/proposer` | LLM proposer and the hybrid arm | 5, 7 | M2 |
| 9 | `feat/analysis` | Aggregation, bootstrap, figures | 6 | M3 |
| 10 | `feat/icl` | ICL baselines | 6, 7 | M3 |
| 11 | `feat/smc` | Tempered-SMC control | 3, 4 | M3 |
| 12 | `exp/chembench` | Frozen configs, full runs, results write-up | 8, 9, 10, 11 | M4 |

### 1. `feat/fit` (done)

- **Scope.** `mda/fit.py`, as built:
  - `sigma(y)`: the whitening scale $\max(0.03\lvert y\rvert, 0.01)$.
  - `residuals(y, y_hat, residual)`: the residuals whose squares sum to the RSS, `"whitened"` by default or `"log1p"` (D1).
  - `fit(h, X, y, *, rng, n_starts, warm_start, residual, f)` → `FitResult(theta, rss, y_hat, ok)`, using bounded multistart `scipy.optimize.least_squares` (spec §3, D8). A parameter with `low > 0` whose bounds span more than two decades is optimised, and its starts are drawn, in log space. One with `low == high` is held fixed. `f` takes an already compiled hypothesis, so that a refit does not compile it again.
  - `relative_residuals(y, y_hat, c)` (D5).
  - `residual_input_spearman(X, y, y_hat)`: the residual–input Spearman correlations used in the prompt (D6), one per input in the order of `INPUT_NAMES`. A correlation is 0 when the input or the residual is constant, and all seven are NaN when any residual is non-finite.
  - `Model`, the type of a compiled hypothesis, added to `mda/hypothesis.py`.
- **Constants.** $\rho$, $\sigma_{\mathrm{floor}}$, $c$ and the two-decade threshold are module constants of `mda/fit.py` (`RHO`, `SIGMA_FLOOR`, `REL_FLOOR`, `LOG_SPAN`) until `feat/agent` adds `MDAConfig`.
- **Dependencies.** `scipy`, added with `uv add`.
- **Tests.** `tests/test_fit.py` covers the five cases planned for this branch, then the starts and the error handling:
  - Recovers known constants from noiseless Michaelis–Menten and Hill data.
  - Never leaves the bounds.
  - Non-finite predictions give `ok=False` and `rss=inf` instead of raising.
  - Deterministic for a given RNG; a warm start is used when given.
  - The all-zero $D_0$ of the activator worlds (`c67`, `c78`, `c92`) does not crash.
  - Starts for wide bounds are drawn in log space, random starts rescue a stuck warm start, and ties go to the warm start.
  - Linear-space parameters raise no warnings, and caller errors raise instead of returning `ok=False`.
- **Measured.** A seed law fits on 25 points in tens of milliseconds: about 40 ms at the median, with 8 random starts and a warm start.

### 2. `feat/library` (done)

- **Scope.** `mda/library.py`, as built:
  - `SEED_LIBRARY`: a tuple of the 9 `Hypothesis` objects in spec §2, with the D7 bounds as module constants (`KCAT`, `K_MM`, `HILL_N`, `EA`).
  - `SEED_BY_WORLD`: a read-only mapping from each canonical world ID to its seed, in the order of `canonical_worlds()`.
  - The parameter names are the vendored oracle's, so fitted values compare with a world's constants by name.
- **Tests.** `tests/test_library.py`:
  - All 9 compile, and their names and expressions are unique.
  - Each canonical world's `v0`/medium constants fall within the bounds. This checks the bounds; it is not used to tune them.
  - Each seed at its world's constants reproduces `World.truth`.
  - Fitting each seed to 25 noiseless `World.truth` points (8 random starts) gives RMSLE < 0.05 on `heldout(42, 1000)`.
- **Done when** the fixed-library arm can represent all 9 canonical worlds.

### 3. `feat/posterior` (done)

- **Scope.** As built:
  - `mda/pool.py`:
    - `Candidate`: a hypothesis, its compiled function, its dedupe key, its `origin` (the enum `Origin`: `SEED`, `INIT` or `EXPAND`), `added_at` (the number of observations $B$ when it was added), its fit, and `fit_err`, the in-sample median relative residual of that fit. `n_free` is $C_m$: the parameters with `low < high` (D20).
    - `Pool.add(hypotheses, origin, added_at=B)`: drops exact-expression duplicates (D13), both of the pool and within the batch, and returns the candidates it added. They have no fit until the next `refit`.
    - `Pool.refit(X, y, rng=...)`: fits every candidate to all the data, with 8 random starts and a warm start from the candidate's previous estimate (D8).
    - `Pool.keep(indices)` removes the other candidates. The per-model arrays `rss`, `n_free` and `fit_err` follow the pool order; `rss` and `fit_err` raise while a candidate has no fit.
  - `mda/posterior.py`, which works on per-model arrays and does not import `Pool`:
    - `bic_profiled` with the RSS floor (D2), `bic_fixed`, and `log_prior(n_params, lam)`.
    - `log_weights` and `weights`: the softmax of $-\mathrm{BIC}_m / 2 + \log p(m)$ and its logarithm (spec §3).
    - `prune_top(log_w, n)` returns the indices of the kept models, in pool order, and their renormalised log weights. It ranks log weights because a weight far behind the best underflows to exactly 0, which is routine under the fixed-scale BIC. Ranking the weights would then drop models by pool position, the newest proposals first, instead of by fit.
    - `adaptive_cap(w, fit_err)` returns 6 or 24 (spec §4).
  - `normalise_expr` in `mda/hypothesis.py`: the dedupe key. It shares the comment and whitespace handling of `compile_hypothesis`.
  - Models with non-finite RSS get BIC = inf and weight 0, and are pruned first. When no model has a finite RSS, every weight is 0.
- **Constants.** `N_STARTS` in `mda/pool.py`, and `RSS_FLOOR`, `TAU_C`, `TAU_FIT`, `N_MIN` and `N_MAX` in `mda/posterior.py`, are module constants until `feat/agent` adds `MDAConfig`. The functions already take them as arguments.
- **Tests.** `tests/test_posterior.py`, `tests/test_pool.py` and the `normalise_expr` cases in `tests/test_hypothesis.py` cover the six cases planned for this branch, then the pool's bookkeeping:
  - BIC values against hand computation.
  - Weights sum to 1, and their order is consistent with BIC.
  - $\mathrm{RSS} = 0$ does not produce NaN: exact fits tie, and the parameter count decides between them.
  - Pruning keeps the largest weights. Non-finite models go first, even among weights that underflow to 0.
  - The cap thresholds behave correctly just above and below 0.9 and 0.05.
  - Dedupe ignores formatting but treats a renamed parameter as a new expression.
  - `n_free` leaves out fixed parameters, and a law with one held fixed is ahead of the same law with it free by $\sqrt{N}$.
  - `refit` is deterministic for a given RNG, warm-starts from the previous estimate, and leaves the pool unchanged when a fit raises.
  - On 25 observations of `c1_competitive_inhibition`, the seed pool's weights match Alg. 9 written out per model for both BIC variants, the true seed has weight above 0.99, and the cap drops to 6.
- **Measured.** Refitting the 9 seeds on 25 points takes about half a second.
- **Done when** `Pool.refit(D)` followed by `weights` reproduces Alg. 9 for both BIC variants.

### 4. `feat/design` (done)

- **Scope.** `mda/design.py`, as built:
  - `between_model_variance(U, weights, predictors, *, space)`: $V(a)$ for a batch of unit-box designs (spec §6). It maps `U` to natural units with `from_unit` and compares the predictions on the raw rate by default, or on $\log(1 + y)$ (D9). It takes the weights and one predictor per model, a callable from conditions to rates, and does not import `Pool`, so the SMC control can pass predictors that average over particles. Models with weight 0 are skipped. $V$ is NaN where it is undefined: where a model with positive weight predicts a non-finite value, or where $V$ overflows. An unknown `space` raises `ValueError` before any model is evaluated.
  - `cma_design(weights, predictors, *, rng, n_evals=48, sigma0=0.3, space)` returns the next condition in natural units (D10). pycma runs with its termination criteria ignored, so it always makes exactly `n_evals` evaluations; with the default population of 9, the sixth generation is cut to 3 points and not told to pycma. It returns the best point evaluated, ties going to the earliest, and never a NaN point. When $V$ is NaN at every point evaluated, it raises `DesignError`. pycma samples through `rng`, so it neither reads nor reseeds the global NumPy random state.
  - `random_design(rng)`: uniform in the unit box, the distribution of `World.heldout`.
  - Building the predictors from a pool is left to `feat/agent`: `lambda X, c=c: c.f(c.fit.theta, X)` for each candidate `c`.
- **Constants.** `N_EVALS` and `SIGMA0` are module constants of `mda/design.py` until `feat/agent` adds `MDAConfig`. The functions already take them as arguments.
- **Dependencies.** `cma`, added with `uv add`.
- **Tests.** `tests/test_design.py` covers the four cases planned for this branch, then the edge cases:
  - $V$ matches hand computation in both spaces. $V = 0$ when all models agree or one model has weight 1, and $V \ge 0$ otherwise.
  - Exactly `n_evals` objective evaluations, also when that is not a multiple of the population size and when $V$ is flat, with every point inside the bounds.
  - On a toy two-model pool that disagrees only at high $C_A$, the design moves to high $C_A$.
  - On two seeds that both explain $D_0$ of `c1_competitive_inhibition`, the design beats the best of 48 uniform points in all of 10 seeds, with a median $V$ ratio of 4.4. This is the test that tells a working search from a broken one: 48 uniform points would pass the toy test.
  - pycma's own proposals stay inside the unit box, the best point evaluated is the one returned, and ties go to the earliest.
  - Deterministic for a given seed, and independent of the global random state, which it leaves unchanged.
  - Zero-weight models are never evaluated. Points where $V$ is undefined are never chosen, and the search retreats from them. When no point is defined, `DesignError` is raised after the full budget.
  - Each of these was checked by breaking `cma_design` on purpose: minimising $V$, never updating pycma, dropping the bounds, returning the final mean, reversing the tie rule, ranking NaN points first, returning a point when none is defined, letting a later generation replace a better earlier point, keeping an overflowing $V$ as infinite, and not validating `space`. At least one test fails for each.
- **Measured.** A design for a 24-model pool of seed laws takes about 10 ms.

### 5. `feat/agent`

- **Scope.**
  - `mda/config.py`: `MDAConfig`, one dataclass with every value in spec §1–6 and D1–D15. Each field has a comment citing the paper or a D-number. It takes over `RHO`, `SIGMA_FLOOR`, `REL_FLOOR` and `LOG_SPAN` from `mda/fit.py`, `N_STARTS` from `mda/pool.py`, and `RSS_FLOOR`, `TAU_C`, `TAU_FIT`, `N_MIN` and `N_MAX` from `mda/posterior.py`. `fit` and `residuals` do not yet accept $\rho$, $\sigma_{\mathrm{floor}}$ or the threshold as arguments, so they change here too.
  - `mda/propose.py`: the `Proposer` protocol, `propose(kind, pool, D, budget_left) -> list[Hypothesis]`, plus `NullProposer`.
  - `mda/agent.py`: `MDAAgent` implementing Alg. 2, with the predictive check (both variants, D5), $R_m$ and $N_e^{\max}$ gating (D4), adaptive cap and pruning, a current-data MAP `get_model`, and a plug-in `get_predictor`. Its design step catches `DesignError` from `cma_design` (spec §6), falls back to `random_design` and records the fallback in the run log.
  - `mda/run.py`: `run_cell(world, seed, config, proposer) -> RunLog` (Alg. 1). It evaluates RMSLE on the 1000-point grid, nMSLE on the 60-point query set (D11) and pass, at every $B$ from 3 to 25. The JSON run log records, per step: $x$, $y$, $e^*$, whether the pool expanded, the cap, and a pool snapshot (expression, $\hat\theta$, RSS, BIC, weight, origin, `added_at`), plus the LLM call IDs.
- **Tests.**
  - With `NullProposer`, the pool never expands.
  - With a fake proposer, the pool expands exactly when $e^* > \tau_e$, respecting $R_m$, $N_e^{\max}$ and the cap.
  - The run log has 23 evaluation points and survives a JSON round trip.
  - A smoke run on `c0_michaelis_menten` with the fixed library passes at $B = 25$.
- **Done when** one cell runs end to end with no API calls.

### 6. `feat/runner`

- **Scope.**
  - The CLI: `mda-impl run --arm {hybrid,fixed,random,fixed-tau,...} --worlds ... --seeds 0,1,2 --out runs/<arm>`.
  - Multiprocessing across cells, resuming by skipping completed cells, and a manifest recording the git SHA, config hash, package versions and model ID.
  - Arm configs in `configs/`, with `runs/` and `cache/` added to `.gitignore`.
- **Tests.**
  - The CLI runs a 2-cell grid.
  - Resume skips completed cells.
  - Each arm config loads.
- **Done when** milestone M1 is reached (below).

### 7. `feat/llm-client`

- **Scope.** `mda/llm.py`:
  - `LLMClient.complete(system, user, *, schema=None) -> Reply(text, usage, stop_reason, request_id)` on the Anthropic SDK: `claude-haiku-5-5` with `output_config={"effort": "medium"}`, plus an optional structured-output schema (D15).
  - An on-disk reply cache (D17) and JSONL transcripts.
  - Usage-to-cost accounting, and a `FakeLLM` for tests.
  - The official `count_tokens` measurement for the proposer prompt, which closes the TODO in [api-cost-estimation.md](./api-cost-estimation.md).
- **Tests.**
  - A cache hit makes no network call.
  - `max_tokens` and `refusal` stop reasons are surfaced.
  - Cost arithmetic.
  - No test calls the real API.
- **Done when** a single real call made by hand succeeds and is cached.

### 8. `feat/proposer`

- **Scope.** `LLMProposer` in `mda/propose.py`:
  - Renders the init and expand prompts. The template follows `chem_propose.txt`: the init prompt has no refining section, and the expand prompt adds the refining section and the budget/phase line, filled in as described in D6.
  - Then `parse_hypotheses`, dedupe against the pool, and a cap at $N$ (12 at init, 6 per expansion), with the retry policy of D14.
  - Wiring for the `hybrid` arm.
- **Tests.**
  - Snapshot tests of the rendered prompts.
  - Parsing, dedupe and capping with `FakeLLM`.
  - Retry on unusable replies, and fallback to the seeds when the initial batch is empty.
- **Pilot.** Run the hybrid arm on about 6 worlds × 1 seed. Record tokens and cost, and update [api-cost-estimation.md](./api-cost-estimation.md) with the measured numbers before the full run.
- **Done when** milestone M2 is reached.

### 9. `feat/analysis`

- **Scope.** `mda/analysis.py` and `mda-impl report`:
  - Load run logs, aggregate (seed mean within world, geometric mean over worlds), 5,000-replicate world-bootstrap CIs, paired ratios (D19), and pass rates over all 171 cells (spec §7).
  - Figures: nMSLE against $B$ per arm (Fig. 2a), the posterior storyboard of one cell (Fig. 2b), fixed vs hybrid (Fig. 6), and CMA-ES vs random for each backend (Fig. 7).
- **Dependencies.** `uv add matplotlib`, or a `--group analysis` dependency group.
- **Tests.**
  - Aggregation on a toy set of run logs matches hand computation.
  - The bootstrap is reproducible for a given seed.
  - Infinite errors are handled as spec §7 describes.

### 10. `feat/icl`

- **Scope.** `mda/icl.py` and the ICL prompts in `prompts/` (D16):
  - **ICL (+MDA design)** replays the MDA run logs. For every $B$, it makes one request predicting the 60 query rates. The histories are fixed, so this goes through the Message Batches API at half price.
  - **ICL (+LLM design)** starts from $D_0$, lets the LLM choose 22 conditions, and predicts at every $B$.
- **Tests.**
  - Prompt rendering.
  - Parsing predictions and design replies, with the retry and fallback paths, using `FakeLLM`.

### 11. `feat/smc`

- **Scope.** `mda/smc.py`:
  - Adaptive-tempering SMC per Alg. 6, with 100 particles per structure, random-walk Metropolis that rejects out-of-bounds proposals, and the D18 settings.
  - $p(m) \propto e^{-2C_m}$.
  - $\bar\mu_m(a)$ averaged over particles in $V(a)$.
  - A `--arm smc` config.
- **Tests.**
  - On a one-parameter toy model, $\log Z$ matches numerical quadrature.
  - The ESS stays at or above $\eta N$ after each rung.
  - Particles never leave the bounds.

### 12. `exp/chembench`

- **Scope.** This branch holds no library code. It records:
  - the frozen configs for every arm;
  - the commands used;
  - summary CSVs and figures;
  - `docs/results-chembench.md`, which compares our numbers side by side with the paper's (spec §8) and explains the deviations: the LLM model, the cohort, and every D-default that matters.
- **Raw data.** Raw run logs stay out of git.

## Milestones

| Milestone | Reached when | Check against the paper |
| --- | --- | --- |
| M1: MDA without an LLM | The fixed-library arm runs on all 171 cells with no API cost | Every canonical-world cell passes at $B = 15$ (Fig. 6) |
| M2: Hybrid MDA | The pilot's cost is measured, then all 171 cells run | Pass rate near 74.9% at $B = 20$ and 78.4% at $B = 25$. The pool trajectory looks like Fig. 2b: expansions are rare and the cap drops to 6 once the fit is adequate |
| M3: Baselines and controls | The random-design, fixed-scale-BIC, SMC and ICL arms run | Hybrid minus fixed pass rate at $B = 15$ is about +33 points. The CMA-ES/random nMSE ratio is below 1 under both backends. ICL is worse than MDA |
| M4: Write-up | `docs/results-chembench.md` is merged | Every number in spec §8 is reproduced, or the difference is explained |

The paper's numbers come from GPT-5.6-Luna, and ours from Claude Haiku 5.5, so matching the trends and orders of magnitude is the realistic target. Matching to the last digit is not.

## Conventions

- **Branches.** `feat/<name>` for library code, `dev/<name>` for tooling and docs, `exp/<name>` for experiment runs and results. Each branch is cut from `main` after its dependencies merge, and becomes one PR into `main` with CI green (pre-commit, ty, pytest).
- **Commits.** Messages start with `add:`, `update:`, `fix:` or `rm:`, as in the existing history.
- **Hyperparameters.** Every hyperparameter lives in `MDAConfig`, never inline, with a comment citing the paper location or a D-number from the spec. When a default changes, update the spec in the same PR.
- **Tests.** Tests never call the Claude API; use `FakeLLM` or a fake `Proposer`. Array-typed functions use the jaxtyping aliases in `mda.arrays`, so pytest checks shapes at run time.
- **Dependencies.** Add them with `uv add`, in the branch that first needs them: `scipy` (`feat/fit`, added), `cma` (`feat/design`, added), `matplotlib` (`feat/analysis`).
- **Vendored code.** `src/mda/_vendor` stays identical to upstream.

## Cost and compute

- **API.** The hybrid arm makes at most 23 proposer calls per cell (1 at initialisation and up to 22 expansions, D4), over 171 cells. The estimate in [api-cost-estimation.md](./api-cost-estimation.md) stands until the M2 pilot replaces it with measured usage. Only the fixed-library arm makes no calls. The random-design, fixed-scale-BIC and SMC controls make their own expansion calls; the paper's fixed-scale arms reuse the paired arm's initialisation reply (Fig. 7 caption) instead of making a new call.
- **CPU.** Each update refits at most 24 models with 9 starts each (8 random starts and a warm start) on at most 25 points, and runs 48 CMA-ES evaluations of $V(a)$. A seed law fits in about 40 ms, so refitting a full pool takes about a second, and longer when the laws have more parameters. This is cheap, so parallelise across cells, not within them. SMC costs about 100 particles × rungs × moves per model per update, which is why the paper limits it to a 12-world cohort (D12).

## Future to-dos

These are not scheduled in the branch plan, and no milestone needs them.

### Looser duplicate matching

- **Now.** `normalise_expr` gives two expressions the same dedupe key when they differ only in whitespace, comments, redundant parentheses or the spelling of a number (D13). Parentheses that change the grouping (`a*(b*c)` against `a*b*c`) and a different operand order (`Km + C_A` against `C_A + Km`) make a new expression, so the weight of one law can split between such copies (spec §11, "Near-duplicates").
- **Change.** Build the key with a recursive pass over the AST, in place of `ast.unparse`:
  - Flatten nested `+` chains into one sum and nested `*` chains into one product. This removes the grouping differences.
  - Sort the operands of each sum and product by their own keys. This removes the ordering differences.
  - Leave `**` and function calls as they are, and canonicalise only their arguments.
- **Options.** How far to take `-` and `/` decides the size of the change:
  - Without them, `a - b + c` and `c + a - b` stay distinct, as do `a*b/c` and `a/c*b`.
  - With them, a subtraction becomes a sum of signed terms, and a division a product of numerator and denominator factors.
  - Pulling the sign out of products also merges `-a*b`, `-(a*b)` and `a*(-b)`.
- **Cost.** An estimate, not prototyped: about 30 lines without `-` and `/`, and 40 to 60 with them, plus tests. Only `normalise_expr` changes. `Pool` treats the key as an opaque string, and each candidate still evaluates its own expression, so floating-point differences between the merged forms do not matter.
- **Not covered.** Both of these cost considerably more:
  - Algebraic rewrites, such as `Km*(1 + C_I/Ki)` against `Km + Km*C_I/Ki`, or `C_A*C_A` against `C_A**2`. They need sympy, or a numerical comparison of the two functions at random points.
  - Renamed parameters. Once operands are sorted, the canonical order depends on the names, so renaming needs its own handling.
- **Spec.** The paper matches exact expressions (D13), so adopting this is a deviation from it. Update D13 in the same PR.
