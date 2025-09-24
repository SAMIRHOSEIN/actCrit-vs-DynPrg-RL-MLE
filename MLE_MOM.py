# %%
# Final version to find the best match for alpha
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln,psi # gammaln = log gamma function, psi = digamma (derivative of log gamma function)

def prepare_prob_rows(X, eps=1e-8):
    """Ensure each row is strictly positive and sums to 1."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array (N, K).")
    X = np.clip(X, eps, None) # Ensures all entries are at least eps to avoid zeros due to log(0)
    X /= X.sum(axis=1, keepdims=True) # Renormalize each row for rows whose CS percentages don’t sum to 1 (sum of rows in original data may not be exactly 1)
    return X


def dirichlet_nll_and_grad(alpha, S, N):
    """
    Negative log-likelihood and gradient for Dirichlet given sufficient stats.

    Parameters
    ----------
    alpha : (K,) array, alpha_k > 0
    S     : (K,) array, S_k = sum_i log X[i,k]
    N     : int, number of rows

    Returns
    -------
    nll : float
    grad : (K,) array
    """
    alpha = np.asarray(alpha, dtype=float)

    # If, during optimization, the algorithm tries any α value that's zero or negative, the log-likelihood and its gradient are undefined or invalid (would cause math errors or return NaNs/Infs).
    # By returning np.inf for the objective (negative log-likelihood) and a vector of NaNs for the gradient, we force the optimizer to stay within the valid region (α_k>0).
    if np.any(alpha <= 0):
        return np.inf, np.full_like(alpha, np.nan)

    a0 = alpha.sum()

    # log-likelihood
    ll = N * (gammaln(a0) - np.sum(gammaln(alpha))) + np.dot(alpha - 1.0, S)
    # gradient of log-likelihood, why we defined it?
    # Optimization algorithms (like L-BFGS-B) use the gradient (the derivative with respect to alpha) to find the minimum much faster and more accurately.
    g = N * (psi(a0) - psi(alpha)) + S # Passing the analytic gradient to the optimizer makes it faster and more reliable(cause we use method="L-BFGS-B").
    # return negative for minimizer
    return -ll, -g


def method_of_moments_init(X, var_min=1e-12, alpha0_min=1e-2, alpha0_max=1e6):

    """
    MoM initialization:
    1) mu_k = mean of X[:,k]
    2) var_k = var of X[:,k]
    3) alpha0 ≈ median_k( mu_k*(1-mu_k)/var_k - 1 ), clipped
    4) alpha_k = mu_k * alpha0
    """
    mu = X.mean(axis=0) # for each column, mean of each condition states
    var = X.var(axis=0, ddof=1) # for each column
    var = np.clip(var, var_min, None) # avoid div by zero

    # Only use coordinates with 0 < mu < 1 (avoid mu=0 or 1 degeneracy)
    # we need to check because:if a column has mu=0 or mu=1, or mu=infinity -> cand = -1 and this isn't valid for alpha0.
    valid = np.isfinite(mu) & np.isfinite(var) & (mu > 0) & (mu < 1)
    cand = mu[valid] * (1 - mu[valid]) / var[valid] - 1.0

    # keep finite candidates
    cand = cand[np.isfinite(cand)] # we need to check this because if var is 0 or NaN -> cand = -1 and this is not valid for alpha0.

    
    if cand.size == 0: # if every condition state wasn't valid(e.g., all 0s), we consider alpha0=1.0
        alpha0 = 1.0 # Fallback if everything is degenerate or NaN
    else:
        # calculate the median of all columns(this is one scalar)
        alpha0 = float(np.median(cand))
        alpha0 = float(np.clip(alpha0, alpha0_min, alpha0_max))

    alpha = mu * alpha0   # This gives initial alpha for the whole dataset and for each condition state(alpha is multiplication of mean and median).
    # Ensure strictly positive alphas, even if some mu are 0/1
    alpha = np.clip(alpha, 1e-3, None) 
    return alpha




def fit_dirichlet_mle_lbfgsb(X, x0=None, tol=1e-8, maxiter=1000, verbose=True):
    """
    Fit Dirichlet(alpha) by MLE using L-BFGS-B with analytic gradient.
    we use L-BFGS-B because it handles bound constraints (alpha_k > 0, cause dirichlet is defined for positive value) and is efficient for large problems.
    1) Prepare the data: ensure each row is a valid probability vector.
    2) Compute sufficient statistics: precompute sum of log(X) for efficiency.
    3) Initialize alpha: use method-of-moments if no initial guess is provided.
    4) Define the objective function: negative log-likelihood and its gradient.
    5) Optimize using L-BFGS-B: minimize the negative log-likelihood with bounds on alpha.
    6) check convergence and return results.
    7) Returns: estimated alpha and optimization result object.
    """
    X = prepare_prob_rows(X) # ensure valid input
    N, K = X.shape

    # sufficient statistics - precompute for efficiency
    S = np.sum(np.log(X), axis=0)

    if x0 is None:
        x0 = method_of_moments_init(X)

    bounds = [(1e-9, None)] * K # alpha_k > 0, make sure alphas for each condition state is >0

    def fun(alpha):
        nll, grad = dirichlet_nll_and_grad(alpha, S, N)
        return nll, grad

    res = minimize(fun=fun, x0=x0, method="L-BFGS-B",
                   jac=True, bounds=bounds,
                   options={"disp": verbose, "maxiter": maxiter, "ftol": tol})
    if not res.success and verbose:
        print("Warning: optimization did not converge:", res.message)

    alpha_hat = res.x
    return alpha_hat, res



# np.random.seed(42)
# X = np.random.dirichlet([2, 0.5, 0.5, 0.1], size=100)  # simulate some "realistic" sparse data

file_path = './Datainfobridge/NBEExport_September_18_2025_12_51_19.txt'
# Try to read as CSV (if it fails, try with sep="\t" for tab-delimited)
try:
    df = pd.read_csv(file_path)
except Exception:
    df = pd.read_csv(file_path, sep="\t")
    
cols = ["CS_PERCENT_1", "CS_PERCENT_2", "CS_PERCENT_3", "CS_PERCENT_4"]
X = df[cols].values


alpha_hat, res = fit_dirichlet_mle_lbfgsb(X)
print("Estimated alpha:", alpha_hat)
print("Mean (alpha/sum):", alpha_hat / alpha_hat.sum())

# %%
