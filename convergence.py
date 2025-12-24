#%%
# convergence.py
"""
Manual convergence workflow (DP-only)

1) Run DPvsPPO.py multiple times with different ELE_DP_N_EPISODES (e.g., 10, 100, 1000, 10000, ...).
2) Copy the printed DP summary lines into RAW_LINES below.
3) Run: python convergence.py

This script will:
- Parse each run (N, mean, 95% CI, SD).
- Print a clean table and explicitly state whether each N meets the tolerance rule.
- Choose the "best episode count":
      best N = smallest N such that CI_half_width <= CI_REL_TOL * |mean|
  If none meets it, best N = largest N we tested (most precise available).
- Plot mean ± 95% CI as a function of N.

Tolerance rule (precision criterion)
-----------------------------------
Let x̄ be the estimated mean episode return from N episodes, and let hw be the 95% CI half-width.

We say the estimate is "precise enough" if:
    hw <= CI_REL_TOL * |x̄|

This means:
"At 95% confidence, the true expected return is within ±(CI_REL_TOL * |x̄|) of the estimated mean."

Notes
-----
- Your returns are negative (cost-shaped reward). "Better" usually means less negative (higher).
- This script does NOT compare DP with PPO; it only analyzes DP convergence across N.
"""

import re
from dataclasses import dataclass
from typing import List
import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1) Paste the copied lines here
# =========================
RAW_LINES = [
    "DP evaluation (episode return for 10 episodes): mean=-31982.6760, 95% CI=[-65041.0879, 1075.7360], SD=53336.6723, N=10",
    "DP evaluation (episode return for 100 episodes): mean=-52222.7095, 95% CI=[-63892.0905, -40553.3284], SD=59537.6584, N=100",
    "DP evaluation (episode return for 1000 episodes): mean=-37504.0656, 95% CI=[-40720.7094, -34287.4217], SD=51897.5564, N=1000",
    "DP evaluation (episode return for 10000 episodes): mean=-35504.8808, 95% CI=[-36479.1171, -34530.6445], SD=49705.9321, N=10000",
    "DP evaluation (episode return for 100000 episodes): mean=-36042.3372, 95% CI=[-36353.6719, -35731.0025], SD=50230.9603, N=100000"
]


# =========================
# 2) Settings
# =========================
CI_REL_TOL = 0.05   # stop criterion: CI_half_width <= CI_REL_TOL * |mean|
CI_ABS_TOL = 1e-6   # numeric safety if mean is extremely close to 0 (rare for your case)


# =========================
# 3) Data model and parsing
# =========================
@dataclass(frozen=True)
class RunStat:
    n: int
    mean: float
    ci_low: float
    ci_high: float
    sd: float

    @property
    def half_width(self) -> float:
        return 0.5 * (self.ci_high - self.ci_low)

    @property
    def rel_half_width(self) -> float:
        denom = max(abs(self.mean), CI_ABS_TOL)
        return self.half_width / denom

    @property
    def meets_tol(self) -> bool:
        denom = max(abs(self.mean), CI_ABS_TOL)
        return self.half_width <= CI_REL_TOL * denom


LINE_RE = re.compile(
    r"""
    ^\s*
    DP\s+evaluation.*?
    mean\s*=\s*(?P<mean>[-+]?[\d\.eE]+)\s*,\s*
    95%\s*CI\s*=\s*\[\s*(?P<lo>[-+]?[\d\.eE]+)\s*,\s*(?P<hi>[-+]?[\d\.eE]+)\s*\]\s*,\s*
    SD\s*=\s*(?P<sd>[-+]?[\d\.eE]+)\s*,\s*
    N\s*=\s*(?P<n>\d+)
    \s*$
    """,
    re.VERBOSE
)


def parse_lines(raw_lines: List[str]) -> List[RunStat]:
    stats: List[RunStat] = []
    for line in raw_lines:
        m = LINE_RE.match(line)
        if not m:
            raise ValueError(
                f"Could not parse this line:\n{line}\n\n"
                f"Expected something like:\n"
                f"DP evaluation (...): mean=-1200.0, 95% CI=[-1300.0, -1100.0], SD=200.0, N=100"
            )
        stats.append(RunStat(
            n=int(m.group("n")),
            mean=float(m.group("mean")),
            ci_low=float(m.group("lo")),
            ci_high=float(m.group("hi")),
            sd=float(m.group("sd")),
        ))

    # sort by N increasing (important for convergence plot + best N selection)
    stats.sort(key=lambda s: s.n)
    return stats


# =========================
# 4) Best-N selection
# =========================
def pick_best_n(dp_stats: List[RunStat]) -> RunStat:
    """
    Best N = smallest N that meets the tolerance:
        CI_half_width <= CI_REL_TOL * |mean|
    If none meets it, return the largest N tested (most precise among your runs).
    """
    for s in dp_stats:
        if s.meets_tol:
            return s
    return dp_stats[-1]


# =========================
# 5) Plotting
# =========================
def plot_convergence(dp_stats: List[RunStat]):
    n = np.array([s.n for s in dp_stats], dtype=float)
    mean = np.array([s.mean for s in dp_stats], dtype=float)
    lo = np.array([s.ci_low for s in dp_stats], dtype=float)
    hi = np.array([s.ci_high for s in dp_stats], dtype=float)

    plt.figure()
    plt.plot(n, mean, marker="o", linewidth=1, label="DP mean")
    plt.fill_between(n, lo, hi, alpha=0.2, label="DP 95% CI")

    plt.xlabel("Number of evaluation episodes (N)")
    plt.ylabel("Mean episode return")
    plt.title("DP convergence: mean ± 95% CI (stop when CI half-width ≤ "f"{CI_REL_TOL:.2%} × |mean|)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    if not RAW_LINES:
        raise SystemExit(
            "RAW_LINES is empty.\n"
            "Run DPvsPPO.py for different ELE_DP_N_EPISODES, copy the printed summary lines,\n"
            "paste them into RAW_LINES in convergence.py, then rerun convergence.py."
        )

    dp_stats = parse_lines(RAW_LINES)

    print("\nParsed DP runs:")
    for s in dp_stats:
        status = f"(meets tol {CI_REL_TOL:.2%})" if s.meets_tol else f"(does NOT meet tol {CI_REL_TOL:.2%})"
        print(
            f"  N={s.n:>7d}  mean={s.mean: .4f}  CI=[{s.ci_low: .4f}, {s.ci_high: .4f}]"
            f"  half={s.half_width: .4f}  rel_half={s.rel_half_width: .4f}  {status}"
        )

    best = pick_best_n(dp_stats)
    best_status = f"(meets tol {CI_REL_TOL:.2%})" if best.meets_tol else f"(does NOT meet tol {CI_REL_TOL:.2%}; best available)"
    print("\nBest episode count (DP):")
    print(
        f"  N={best.n}  mean={best.mean:.4f}  CI=[{best.ci_low:.4f}, {best.ci_high:.4f}]  {best_status}"
    )

    plot_convergence(dp_stats)


if __name__ == "__main__":
    main()
