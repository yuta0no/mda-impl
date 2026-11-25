"""The seed library: the nine canonical ChemBench rate laws (spec §2, App. B.2).

Each seed has the functional form of one canonical benchmark world (`mda.bench.canonical_worlds()`).
The paper gives no parameter bounds, so they are set from physical scales and the input box, not
from the worlds' constants (D7). Parameter names match the vendored oracle's.
"""

from collections.abc import Mapping
from types import MappingProxyType

from mda.hypothesis import Hypothesis, Param

# Parameter bounds (D7)
KCAT = (1e-3, 1e3)  # turnover number
K_MM = (1e-3, 1e3)  # Km, Ki, Kp, KmA, KmB, K_half, in mM
HILL_N = (0.1, 8.0)  # Hill coefficient
EA = (1e3, 2e5)  # activation energy, in J/mol


def _kcat(name: str = "kcat") -> Param:
    return Param(name, *KCAT)


def _k(name: str) -> Param:
    return Param(name, *K_MM)


SEED_BY_WORLD: Mapping[str, Hypothesis] = MappingProxyType(
    {
        "c0_michaelis_menten": Hypothesis(
            "michaelis_menten",
            "kcat*Enz*C_A/(Km + C_A)",
            (_kcat(), _k("Km")),
        ),
        "c1_competitive_inhibition": Hypothesis(
            "competitive_inhibition",
            "kcat*Enz*C_A/(Km*(1 + C_I/Ki) + C_A)",
            (_kcat(), _k("Km"), _k("Ki")),
        ),
        "c2_product_inhibition": Hypothesis(
            "product_inhibition",
            "kcat*Enz*C_A/(Km*(1 + C_P/Kp) + C_A)",
            (_kcat(), _k("Km"), _k("Kp")),
        ),
        "c3_arrhenius_temperature": Hypothesis(
            "arrhenius",
            "kcat_ref*exp(-Ea/8.314*(1/T - 1/310))*Enz*C_A/(Km + C_A)",
            (_kcat("kcat_ref"), Param("Ea", *EA), _k("Km")),
        ),
        "c5_pingpong_bisubstrate": Hypothesis(
            "pingpong_bisubstrate",
            "kcat*Enz*C_A*C_B/(KmA*C_B + KmB*C_A + C_A*C_B)",
            (_kcat(), _k("KmA"), _k("KmB")),
        ),
        "c6_uncompetitive_inhibition": Hypothesis(
            "uncompetitive_inhibition",
            "kcat*Enz*C_A/(Km + C_A*(1 + C_I/Ki))",
            (_kcat(), _k("Km"), _k("Ki")),
        ),
        "c7_substrate_inhibition": Hypothesis(
            "substrate_inhibition",
            "kcat*Enz*C_A/(Km + C_A + C_A**2/Ki)",
            (_kcat(), _k("Km"), _k("Ki")),
        ),
        "c8_hill_cooperativity": Hypothesis(
            "hill",
            "kcat*Enz*C_A**n/(K_half**n + C_A**n)",
            (_kcat(), _k("K_half"), Param("n", *HILL_N)),
        ),
        "c9_noncompetitive_inhibition": Hypothesis(
            "noncompetitive_inhibition",
            "kcat*Enz*C_A/((1 + C_I/Ki)*(Km + C_A))",
            (_kcat(), _k("Km"), _k("Ki")),
        ),
    }
)
"""Each canonical world's ID mapped to the seed with its functional form."""

SEED_LIBRARY: tuple[Hypothesis, ...] = tuple(SEED_BY_WORLD.values())
"""The nine seed laws (spec §2), in the order of `mda.bench.canonical_worlds()`."""
