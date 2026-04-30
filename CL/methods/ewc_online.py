import numpy as np
import tensorflow as tf
from typing import List

from CL.methods.ewc import EWC_SAC


class EWCOnline_SAC(EWC_SAC):
    """Online EWC with an exponential moving average of the Fisher diagonal.

    The original EWC of Kirkpatrick et al. (2017) accumulates the
    per-task Fisher diagonals additively, so the regularisation
    constraint grows monotonically with the number of tasks. Schwarz
    et al. (2018, "Progress & Compress") replace the running sum with
    an exponential moving average

        F_total <- gamma * F_old + (1 - gamma) * F_new,    gamma in [0, 1]

    which keeps the constraint magnitude bounded across long
    sequences. In the low-budget regime studied in this dissertation,
    where the per-task Fisher estimate is itself dominated by
    optimisation noise (the "Fisher on noise" effect), the EMA acts as
    a low-pass filter on that noise as well, which is the reason this
    variant is expected to recover ground against the unregularised
    Fine-Tuning control.

    Arguments inherited unchanged from EWC_SAC; the new behaviour is
    controlled by `ewc_gamma` (default 0.95).
    """

    def __init__(self, cl_reg_coef: float = 1.0, regularize_critic: bool = False,
                 ewc_gamma: float = 0.95, **kwargs) -> None:
        super().__init__(cl_reg_coef=cl_reg_coef,
                         regularize_critic=regularize_critic, **kwargs)
        if not 0.0 <= ewc_gamma <= 1.0:
            raise ValueError(f"ewc_gamma must be in [0, 1]; got {ewc_gamma}")
        self.ewc_gamma = float(ewc_gamma)
        self._fisher_initialised = False

    def _merge_weights(self, new_weights: List[tf.Variable]) -> None:
        """Replace the additive Fisher accumulation in `Regularization_SAC`
        with an exponential moving average. The first merge after the
        first task installs the new Fisher directly via the same
        broadcast addition the baseline uses (`reg_weights` is
        zero-initialised at this point); subsequent merges decay the
        running estimate by `gamma` and add `(1 - gamma)` of the new
        Fisher."""
        if not self._fisher_initialised:
            merged_weights = list(
                old_reg + new_reg
                for old_reg, new_reg in zip(self.reg_weights, new_weights)
            )
            self._fisher_initialised = True
        else:
            g = self.ewc_gamma
            merged_weights = list(
                g * old_reg + (1.0 - g) * new_reg
                for old_reg, new_reg in zip(self.reg_weights, new_weights)
            )

        for old_weight, new_weight in zip(self.reg_weights, merged_weights):
            old_weight.assign(new_weight)
