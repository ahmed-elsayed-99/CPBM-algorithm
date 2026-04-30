import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


class StratumGradient:

    def __init__(self, onset_pct: float = 0.10, random_state: int = 42):
        self.onset_pct = onset_pct
        self.random_state = random_state
        self.bass_params_: Dict[int, Dict[str, float]] = {}
        self.tau_: Dict[str, float] = {}
        self.adoption_peaks_: Dict[int, float] = {}

    @staticmethod
    def _bass_ode(y: List[float], t: float, p: float, q: float, M: float) -> List[float]:
        F = y[0] / (M + 1e-12)
        dF = (p + q * F) * (1.0 - F) * M
        return [dF]

    def _bass_cumulative(self, t: np.ndarray, p: float, q: float, M: float) -> np.ndarray:
        y0 = [0.0]
        sol = odeint(self._bass_ode, y0, t, args=(p, q, M), rtol=1e-6, atol=1e-8)
        return sol[:, 0]

    def fit_stratum(
        self,
        time_axis: np.ndarray,
        adoption_counts: np.ndarray,
        M: float,
        stratum_id: int,
    ) -> Dict[str, float]:
        def _model(t, p, q):
            return self._bass_cumulative(t, p, q, M)

        p0 = [0.03, 0.38]
        bounds = ([1e-4, 1e-3], [0.5, 2.0])
        try:
            popt, _ = curve_fit(
                _model, time_axis, adoption_counts,
                p0=p0, bounds=bounds, maxfev=20000,
            )
            params = {"p": popt[0], "q": popt[1], "M": M}
        except RuntimeError:
            params = {"p": 0.03, "q": 0.38, "M": M}

        self.bass_params_[stratum_id] = params
        adoption_rate = np.gradient(self._bass_cumulative(time_axis, params["p"], params["q"], M), time_axis)
        peak_idx = np.argmax(adoption_rate)
        self.adoption_peaks_[stratum_id] = float(time_axis[peak_idx])
        return params

    def fit(
        self,
        stratum_adoption: Dict[int, Tuple[np.ndarray, np.ndarray, float]],
    ) -> "StratumGradient":
        for stratum_id, (time_axis, adoption_counts, M) in stratum_adoption.items():
            self.fit_stratum(time_axis, adoption_counts, M, stratum_id)

        strata_ids = sorted(self.adoption_peaks_.keys(), reverse=True)
        for idx in range(len(strata_ids) - 1):
            upper = strata_ids[idx]
            lower = strata_ids[idx + 1]
            t_peak_upper = self.adoption_peaks_[upper]

            lower_params = self.bass_params_[lower]
            time_axis_ref = list(stratum_adoption[lower])[0] if isinstance(stratum_adoption[lower], tuple) else np.linspace(0, 90, 91)
            time_axis_ref = stratum_adoption[lower][0]
            lower_curve = self._bass_cumulative(
                time_axis_ref,
                lower_params["p"],
                lower_params["q"],
                lower_params["M"],
            )
            onset_threshold = self.onset_pct * lower_params["M"]
            onset_mask = lower_curve >= onset_threshold
            if onset_mask.any():
                t_onset_lower = float(time_axis_ref[np.argmax(onset_mask)])
            else:
                t_onset_lower = float(time_axis_ref[-1])

            tau_key = f"tau_{upper}_{lower}"
            self.tau_[tau_key] = t_onset_lower - t_peak_upper

        return self

    def get_tau(self, upper: int, lower: int) -> float:
        key = f"tau_{upper}_{lower}"
        return self.tau_.get(key, np.nan)

    def predict_adoption(
        self,
        stratum_id: int,
        time_axis: np.ndarray,
    ) -> np.ndarray:
        if stratum_id not in self.bass_params_:
            raise KeyError(f"Stratum {stratum_id} not fitted.")
        p = self.bass_params_[stratum_id]
        return self._bass_cumulative(time_axis, p["p"], p["q"], p["M"])

    def tau_interpretation(self, tau_days: float) -> Dict[str, str]:
        if tau_days < 14:
            return {"category": "viral", "strategy": "simultaneous mass-market launch", "window_weeks": "<2"}
        elif tau_days < 42:
            return {"category": "aspirational", "strategy": "premium launch then mid-tier follow-up", "window_weeks": "2-6"}
        elif tau_days < 112:
            return {"category": "status_good", "strategy": "skim price then planned markdown", "window_weeks": "6-16"}
        else:
            return {"category": "luxury_cascade", "strategy": "separate SKUs per stratum", "window_weeks": ">16"}

    def summary(self) -> pd.DataFrame:
        rows = []
        for key, tau in self.tau_.items():
            parts = key.split("_")
            upper, lower = int(parts[1]), int(parts[2])
            interp = self.tau_interpretation(tau)
            rows.append(
                {
                    "stratum_upper": upper,
                    "stratum_lower": lower,
                    "tau_days": round(tau, 2),
                    "tau_weeks": round(tau / 7, 2),
                    "category": interp["category"],
                    "strategy": interp["strategy"],
                }
            )
        return pd.DataFrame(rows)