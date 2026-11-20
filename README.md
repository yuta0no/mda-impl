# MDA-impl

This repository implements [Model Discovery Agent (MDA)](https://arxiv.org/abs/2608.09696).

## Development

The project is managed with [uv](https://docs.astral.sh/uv/), which also installs the Python version in `.python-version` (3.12) if it is missing. CI tests on Python 3.12 and 3.13.

```sh
uv sync                      # create .venv with the package and the dev tools (pytest, ruff, ty, beartype)
uv tool install pre-commit   # once per machine
pre-commit install           # once per clone: run the hooks on every commit
cp .env.sample .env          # then set ANTHROPIC_API_KEY in .env
```

`.env` is ignored by git. The API key is only needed by code that calls the Claude API, such as `tools/token_count.py`.

### Checks

| Command | What it runs |
| --- | --- |
| `uv run pytest` | The tests. For the calls the tests make into `mda`, they also check jaxtyping shape and dtype annotations at runtime. |
| `uv run ty check` | The static type checker. It reads jaxtyping annotations as plain `np.ndarray`, so it does not check shapes or dtypes. |
| `pre-commit run --all-files` | Every hook: file checks, ruff lint and format, `uv.lock` sync, and ty. |

- Add dependencies with `uv add <package>`, or `uv add --dev <package>` for dev tools, so that `uv.lock` stays in sync. CI installs with `uv sync --locked`, which fails if the lock file is out of date.
- `src/mda/_vendor` is excluded from ruff and ty and kept identical to upstream (see [Upstream code defects](#upstream-code-defects)), so don't edit it.


## API cost

A conservative estimate for the full ChemBench experiment is under $100 with Claude Haiku 5.5 at medium effort.
See [docs/api-cost-estimation.md](./docs/api-cost-estimation.md) for the breakdown.

## Notes

### ChemBench

Findings about ChemBench (ActiveSciBench-Chem) from reading and running the vendored oracle in [./src/mda/_vendor/autoscilab](./src/mda/_vendor/autoscilab).

#### Benchmark behaviour

These are how the benchmark is designed to work, and an implementation has to account for them:

- **Observations are not the true law plus noise.** The oracle multiplies the true rate by a background temperature/pH factor before adding 1% noise, but scores against the true law alone. The factor ranges from -6.5% to +6.3% over the input box and is exactly 1 at the passive design's $T = 310$ K and pH 7. `World.truth` therefore calls the rate function directly.
- **Three worlds start with all-zero observations.** In `c67_allosteric_act`, `c78_allosteric_act_arrhenius` and `c92_allosteric_act_feedback`, $C_I$ is an activator, and the passive set $D_0$ fixes $C_I = 0$, so all three initial rates are exactly 0.
- **Noise is a deterministic function of the condition.** The noise draw is seeded by a hash of the world and the seven inputs, so repeating an experiment returns exactly the same value, and replicate runs of a world see identical observations at identical conditions. Only the method's own randomness differs between the three replicates.
- **The default law version is random.** With `law_version=None`, the constructor default, the oracle picks one of the world's constant sets with the unseeded `random.choice`, so the constants change from process to process. `mda.bench` pins `v0`, which is also what the ChemBench task manifest in the upstream repository (`configs/noise_studies/chembench_llm_autoscilab_noise27.json`) uses for all 27 of its tasks.

#### Upstream code defects

These look like mistakes in the upstream code. The vendored files are kept identical to upstream so that results stay comparable with the MDA paper, so none of them is fixed here:

- **Eight worlds ignore the inhibitor their name promises.** The compound rate function does not apply $C_I$ for ping-pong + competitive (`c25_pingpong_competitive`, `c27_pingpong_competitive_arrhenius`), Hill + uncompetitive (`c34_hill_uncompetitive`, `c46_hill_uncompetitive_arrhenius`) or substrate inhibition + uncompetitive (`c49_sinh_uncompetitive`, `c63_sinh_uncompetitive_arrhenius`), and does not apply $C_P$ for Hill + product (`c36_hill_product`, `c47_hill_product_arrhenius`). The simulated and scored law is the same mechanism without inhibition. Each therefore has the same functional form, with different constants, as another benchmark world: `c25` as `c5`, `c27` as `c23`, `c34` and `c36` as `c8`, `c46` and `c47` as `c37`, `c49` as `c7`, and `c63` as `c52`.
- **Two ground-truth law strings do not match the oracle.** For `c49` and `c63`, `get_ground_truth_law_str()` includes a $C_I$ term that the rate function lacks, so the string does not score as correct against the oracle's own reference (RMSLE 0.80 and 1.31). For the other six worlds above, the string also omits the inhibitor and is consistent with the oracle.
- **`c93_fractal_competitive` does not use competitive inhibition.** Its rate is divided by $1 + (C_I/K_i)^\beta$ independently of $C_A$, so the inhibition is the same at every substrate level instead of vanishing at high $C_A$. The ground-truth law string uses the same form.
- **`c82_metal_act_arrhenius` is excluded on a miscount.** Its only exclusion reason is having more than five free parameters, and the comment lists six, including `Ki`. Its rate never uses `Ki`, so it has five, as do 13 of the 57 benchmark worlds. The miscount comes from `_generate_novel_params`, which samples every parameter in a novel family's template whether or not the world uses it (`_NOVEL_REQUIRED_PARAMS` and `_NOVEL_OPTIONAL_PARAMS` are defined but never used), so the `law_params` of most novel worlds also list unused entries such as `Ea`, `pKa1` and `pKa2`.
- **Several comments disagree with the code.**
  - Two comments in `chembench.py` call `CHEM_BENCHMARK_DOMAINS` "the 54 fair benchmark domains", but the list has 57 entries, matching the 57 worlds in both the AutoSciLab and MDA papers. The same comment block counts 30 pH bell-curve exclusions; there are 39.
  - The `_secondary_effects` docstring says the background factor stays within ±5%; it spans -6.5% to +6.3%.
  - A comment in `make_compound_rate_fn` says ping-pong has no noncompetitive mode, but `c26_pingpong_noncompetitive` and `c29_pingpong_noncompetitive_arrhenius` are ping-pong + noncompetitive, and the multiplier is applied to them.

The following defects are in metadata only. They do not change observations or scores, and nothing in `mda` reads the affected fields:

- **The metadata outputs reveal the noise-free rate.** `run()` computes `C_A_t`, `C_P_t` and `conversion` from the rate before noise is added, so whenever substrate is left over (78% of runs on held-out conditions), $(C_A - C_{A,t})/t$ recovers the noise-free rate, background factor included, to within about $10^{-6}$. A method that reads `metadata["outputs"]` can bypass the 1% noise; `World.observe` returns only `measurement`.
- **Product inhibition is labelled as $C_I$ in the reaction schema.** For `c16_mm_product_arrhenius`, `c36_hill_product`, `c47_hill_product_arrhenius`, `c51_sinh_product` and `c61_sinh_product_arrhenius`, the schema says "inhibitor C_I (product)" although the inhibitor is $C_P$. The schema is only shown with `grammar_mode="domain_specific"`; `mda` uses `"none"`.
- **Metal-activation worlds are marked as consuming $C_B$.** `c73_metal_activation` and `c94_metal_act_competitive` use $C_B$ as a cofactor that is not consumed, but they are in the consume-B set, so the `C_B_t` value in the result metadata is reduced as if it were a substrate. The measured rate is unaffected.
- **`c72_monotonic_ph` omits pH from `relevant_vars`.** The rate depends on pH, but the registry entry does not list it.

### MDA

- TODO


### MDA vs ChemBench

- **The upstream scorer is more lenient.** `ChemBenchOracle.evaluate_law` drops non-finite predictions and points where the truth is 0, and scores absolute values, so a law with the wrong sign scores as perfect. Following the MDA paper, `mda.metrics` counts non-finite predictions as a failure and scores predictions as they are.
- **Four "out-of-library" worlds are in-library in practice.** Because their inhibitor is ignored, `c25` has exactly the canonical ping-pong form (`c5`), `c34` and `c36` the Hill form (`c8`), and `c49` the substrate-inhibition form (`c7`). The paper's fixed-library comparison counts all 48 compound worlds as out-of-library (Fig. 6), but a library of the nine canonical forms can represent these four exactly. By design, `c68_anticoop_hill` is also the Hill form, with $n \approx 0.37$, so whether it is in-library depends on the bounds chosen for $n$.
- **Held-out set.** `World.heldout(42, 1000)` reproduces the test set used by `evaluate_law`.
