# MDA-impl

This repository implements [Model Discovery Agent (MDA)](https://arxiv.org/abs/2608.09696).

## API cost

A conservative estimate for the full ChemBench experiment is under $100 with Claude Haiku 5.5 at medium effort.
See [docs/api-cost-estimation.md](./docs/api-cost-estimation.md) for the breakdown.

## Notes

### ChemBench

Findings about ChemBench (ActiveSciBench-Chem) from reading and running the vendored oracle in [./src/mda/_vendor/autoscilab](./src/mda/_vendor/autoscilab).

#### Benchmark behaviour

These are how the benchmark is designed to work, and an implementation has to account for them:

- **Observations are not the true law plus noise.** The oracle multiplies the true rate by a background temperature/pH factor (about -6% to +6%) before adding 1% noise, but scores against the true law alone. `World.truth` therefore calls the rate function directly.
- **Three worlds start with all-zero observations.** In `c67_allosteric_act`, `c78_allosteric_act_arrhenius` and `c92_allosteric_act_feedback`, $C_I$ is an activator, and the passive set $D_0$ fixes $C_I = 0$, so all three initial rates are exactly 0.

#### Upstream code defects

These look like mistakes in the upstream code. The vendored files are kept identical to upstream so that results stay comparable with the MDA paper, so none of them is fixed here:

- **Six worlds ignore the inhibitor their name promises.** The compound rate function does not apply $C_I$ for ping-pong + competitive (`c25_pingpong_competitive`, `c27_pingpong_competitive_arrhenius`), Hill + uncompetitive (`c34_hill_uncompetitive`, `c46_hill_uncompetitive_arrhenius`) or substrate inhibition + uncompetitive (`c49_sinh_uncompetitive`, `c63_sinh_uncompetitive_arrhenius`). The simulated and scored law is the same mechanism without inhibition.
- **Two ground-truth law strings do not match the oracle.** For `c49` and `c63`, `get_ground_truth_law_str()` includes a $C_I$ term that the rate function lacks, so the string does not score as correct against the oracle's own reference (RMSLE 0.80 and 1.31). For the other four worlds above, the string also omits $C_I$ and is consistent with the oracle.
- **A stale comment says 54 domains.** Two comments in `chembench.py` call `CHEM_BENCHMARK_DOMAINS` "the 54 fair benchmark domains", but the list has 57 entries, matching the 57 worlds in both the AutoSciLab and MDA papers.

### MDA

- TODO


### MDA vs ChemBench

- **The upstream scorer drops invalid points.** `ChemBenchOracle.evaluate_law` masks out non-finite predictions before averaging. Following the MDA paper, `mda.metrics` counts them as a failure instead.
- **Held-out set.** `World.heldout(42, 1000)` reproduces the test set used by `evaluate_law`.
