"""`MDAConfig`: every hyperparameter of the ChemBench experiment, in one place.

Each field cites the paper (arXiv:2608.09696v5) or a default of ours from
`docs/chembench-spec.md` (D1-D20). The other modules take these values as arguments and never
hard-code them; their keyword defaults come from `DEFAULTS`, so a call that names no value uses
the same one as the primary arm.

This module imports nothing from `mda`, so every other module can import it. The option types
(`Residual`, `Space`, ...) are defined here for the same reason.
"""

from dataclasses import asdict, dataclass
from typing import Any, Literal

type Residual = Literal["whitened", "log1p"]  # the residual whose sum of squares is the RSS (D1)
type Bic = Literal["profiled", "fixed"]  # the noise scale of the BIC (spec §3)
type Check = Literal["cached", "prequential"]  # the predictive-check variant (D5)
type Design = Literal["cma", "random"]  # how the next condition is chosen (spec §6)
type Space = Literal["raw", "log1p"]  # the space in which V(a) compares predictions (D9)
# Claude's effort levels (`output_config.effort`). Other providers name and scale theirs
# differently, so a new backend defines its own
type ClaudeEffort = Literal["low", "medium", "high", "xhigh", "max"]


@dataclass(frozen=True)
class MDAConfig:
    """The hyperparameters of one arm. The defaults are the primary arm (spec §8)."""

    # Protocol (spec §1)
    # Passive data D0: one observation per C_A, the other inputs held at `passive_rest` (App. B.2)
    passive_c_a: tuple[float, ...] = (0.03, 0.3, 3.0)
    passive_rest: tuple[float, ...] = (0.0, 1.0, 0.0, 1.0, 310.0, 7.0)  # C_I, C_B, C_P, Enz, T, pH
    n_active: int = 22  # K, the active experiments after D0, one per round (§4.1)

    # Fitting (spec §3)
    rho: float = 0.03  # relative noise scale of the whitening (App. A.6.3, eq. 46)
    sigma_floor: float = 0.01  # absolute floor of the whitening scale (App. A.6.3, eq. 46)
    residual: Residual = "whitened"  # D1
    n_starts: int = 8  # random starts per fit, on top of the warm start (D8)
    log_span: float = 100.0  # bounds spanning more than two decades are fitted in log space (D8)

    # Model weights (spec §3)
    bic: Bic = "profiled"  # the headline configuration; "fixed" is the tau = 1 control (App. B.4)
    rss_floor: float = 1e-12  # floor of RSS / N in the profiled-scale BIC (D2)
    lam: float = 0.0  # structure prior p(m) ∝ exp(-lam C_m): none in the BIC arm (Table 2)

    # Pool (spec §4)
    n_init: int = 12  # N_init, laws requested from the LLM at initialisation (Alg. 11, App. B.2)
    n_new: int = 6  # N_new, laws requested per expansion (Alg. 12, App. B.2)
    n_max: int = 24  # N_max, the pool cap (Alg. 10, App. B.2)
    n_min: int = 6  # N_min, the cap once the fit is adequate (Alg. 10, App. B.2)
    tau_c: float = 0.9  # the cap drops when the largest weight exceeds this (App. B.2)
    tau_fit: float = 0.05  # ... and the MAP model's median relative residual is at most this

    # Update (spec §5)
    tau_e: float = 0.05  # expand when the predictive-check error e* exceeds this (D3)
    r_m: int = 1  # R_m, model-search rounds per update: one gated expansion (D4)
    n_e_max: int | None = None  # N_e^max, the cap on expansions per cell; None is no cap (D4)
    check: Check = "cached"  # what the executed ChemBench agent did (App. A.8, D5)
    backtest: bool = True  # the prequential check also scores the stored data (Alg. 13)
    rel_floor: float = 0.01  # floor c of the relative error (D5)

    # Design (spec §6)
    design: Design = "cma"  # "random" is the random-design control (App. B.4)
    n_evals: int = 48  # objective evaluations per design (App. B.2)
    sigma0: float = 0.3  # initial CMA-ES step size in the unit box (D10)
    design_space: Space = "raw"  # D9

    # Evaluation (spec §7)
    n_query: int = 60  # size of the held-out query set of a (world, seed) cell (D11)

    # LLM (spec §9); read by `mda.llm` and the proposer
    llm_model: str = "claude-haiku-5-5"
    llm_effort: ClaudeEffort = "medium"  # the paper's model ran at medium reasoning effort
    # Room for adaptive thinking plus a reply of up to 12 hypotheses. Thinking tokens count
    # against this cap, and a non-streaming request must stay under the SDK's ~21K-token limit
    llm_max_tokens: int = 16_000
    # The SDK retries 408, 409, 429, 5xx and connection errors with exponential backoff. Its
    # default of 2 retries is short for an unattended run of several hours
    llm_max_retries: int = 5
    max_attempts: int = 3  # attempts per proposer call before going on without candidates (D14)
    structured_output: bool = True  # request the hypotheses JSON schema (D15)
    prompt_phase: bool = True  # include the budget/phase line in the expansion prompt (D6)

    def __post_init__(self) -> None:
        positive = (
            "rho",
            "sigma_floor",
            "log_span",
            "rss_floor",
            "rel_floor",
            "sigma0",
            "n_evals",
            "n_min",
            "n_query",
            "llm_max_tokens",
            "max_attempts",
        )
        non_negative = ("n_active", "n_starts", "n_init", "n_new", "r_m", "tau_e", "tau_fit")
        for name in positive:
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        for name in non_negative:
            if not getattr(self, name) >= 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")
        if self.n_e_max is not None and self.n_e_max < 0:
            raise ValueError(f"n_e_max must be non-negative or None, got {self.n_e_max}")
        if self.n_min > self.n_max:
            raise ValueError(f"n_min ({self.n_min}) exceeds n_max ({self.n_max})")
        if not self.passive_c_a:
            raise ValueError("passive_c_a is empty: the agent needs passive data to start from")
        if len(self.passive_rest) != 6:
            raise ValueError(f"passive_rest holds the 6 inputs after C_A, got {self.passive_rest}")

    def to_dict(self) -> dict[str, Any]:
        """The fields as JSON-compatible values, for run logs and manifests."""
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MDAConfig":
        """The config that `to_dict` wrote. Missing fields take their defaults; unknown ones raise."""
        fields: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}
        return cls(**fields)


DEFAULTS = MDAConfig()
"""The primary arm. Keyword defaults across `mda` read their values from it."""
