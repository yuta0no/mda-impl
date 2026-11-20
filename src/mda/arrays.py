import numpy as np
from jaxtyping import Float

Condition = Float[np.ndarray, "7"]
Conditions = Float[np.ndarray, "n 7"]
Rates = Float[np.ndarray, " n"]
Theta = Float[np.ndarray, " p"]  # Values for the additional parameters specified by LLM's output
