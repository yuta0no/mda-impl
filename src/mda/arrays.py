import numpy as np
from jaxtyping import Float, Int

Condition = Float[np.ndarray, "7"]
Conditions = Float[np.ndarray, "n 7"]
Rates = Float[np.ndarray, " n"]
Theta = Float[np.ndarray, " p"]  # Values for the additional parameters specified by LLM's output
PerModel = Float[np.ndarray, " m"]  # One value per model, in pool order: RSS, BIC, weight, ...
ParamCounts = Int[np.ndarray, " m"]  # Free parameters of each model, those with low < high (D20)
