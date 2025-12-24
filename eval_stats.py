# eval_stats.py
import numpy as np

def mean_and_ci(episode_returns, z: float = 1.96):
    """
    Compute summary statistics for episode returns, including an approximate two-sided confidence interval (CI)
    using a normal (z) critical value.

    Inputs
    ------
    episode_returns : array-like
        1D iterable of floats. Typically: "sum of reward per episode".
    z : float
        Critical value for a two-sided normal CI.
        Examples:
            90% CI  -> z ≈ 1.645
            95% CI  -> z ≈ 1.96   (default)
            99% CI  -> z ≈ 2.576

        If you later want a different confidence level, you change z accordingly.

    Outputs
    -------
    stats : dict
        {
          "n": int,
          "mean": float,
          "sd": float,
          "se": float,
          "ci_low": float,
          "ci_high": float,
          "ci_half_width": float
        }

    Equations used
    --------------
    Given samples x_1, ..., x_n (episode returns):

    Mean:
        mean = x̄ = (1/n) * Σ x_i

    Sample standard deviation (ddof=1):
        sd = sqrt( (1/(n-1)) * Σ (x_i - x̄)^2 )

    Standard error of the mean:
        se = sd / sqrt(n)

    Approx. two-sided CI half-width (normal approximation):
        hw = z * se

    CI bounds:
        CI = [x̄ - hw, x̄ + hw]

    Notes
    -----
    - This is a *normal approximation*. For small n (e.g., < 30), a Student-t critical value is more accurate.
      In your convergence use-case with large n (1000+), z is typically fine.
    """
    x = np.asarray(episode_returns, dtype=np.float64).reshape(-1)
    n = int(x.size)

    # With <2 samples, sd/se are undefined. We return a degenerate CI at the mean.
    if n < 2:
        m = float(x.mean()) if n == 1 else float("nan")
        return {
            "n": n,
            "mean": m,
            "sd": float("nan"),
            "se": float("nan"),
            "ci_low": m,
            "ci_high": m,
            "ci_half_width": float("nan"),
        }

    mean = float(x.mean())
    sd = float(x.std(ddof=1))           # sample SD
    se = float(sd / np.sqrt(n))         # standard error
    hw = float(z * se)                  # CI half-width

    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci_low": float(mean - hw),
        "ci_high": float(mean + hw),
        "ci_half_width": hw,
    }
