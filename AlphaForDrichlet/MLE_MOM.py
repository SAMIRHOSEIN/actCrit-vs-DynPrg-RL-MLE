#%%
import numpy as np
from scipy.special import psi, polygamma  # psi = digamma, polygamma(1, x) = trigamma

def fit_dirichlet_mle(P, tol=1e-6, max_iter=100):
    """
    P: N x K array, each row a probability vector (no zeros! add small epsilon if needed).
    Returns: fitted alpha vector (K,)
    """
    N, K = P.shape

    # Compute sufficient statistics
    logp_mean = np.mean(np.log(P), axis=0)  # shape (K,)
    
    # Method-of-moments initial guess
    mean = np.mean(P, axis=0) #for each column, mean of each condition states
    var = np.var(P, axis=0) #for each column
    alpha0_init = np.median((mean * (1 - mean)) / (var + 1e-8) - 1) # calculate the median of all columns(this is one scalar)
    alpha0_init = max(alpha0_init, 1e-2)
    alpha = mean * alpha0_init  # initial guess of alpha for each column(each condition state) - This gives a single starting vector α(0) for the whole dataset.
    
    for it in range(max_iter):
        alpha0 = np.sum(alpha)
        g = N * (psi(alpha0) - psi(alpha) + logp_mean)  # gradient
        H = -N * polygamma(1, alpha)  # Hessian diagonal
        z = N * polygamma(1, alpha0)  # for Sherman-Morrison

        # Newton update with Sherman-Morrison (solves H*delta = g)
        invH = 1 / H
        b = np.sum(g * invH)
        c = b / (1/z + np.sum(invH))
        delta = (g - c) * invH

        # Line search for positivity
        step = 1.0
        while np.any(alpha - step * delta <= 0):
            step *= 0.5
            if step < 1e-8:
                break
        alpha_new = alpha - step * delta

        # Convergence check
        if np.linalg.norm(alpha_new - alpha) < tol:
            break
        alpha = alpha_new

    return alpha


# Example with random Dirichlet data
np.random.seed(42)
P = np.random.dirichlet([2, 0.5, 0.5, 0.1], size=100)  # simulate some "realistic" sparse data

# Add epsilon if you have zeros
P = np.clip(P, 1e-6, 1.0)
P = (P.T / P.sum(axis=1)).T

alpha_hat = fit_dirichlet_mle(P)
print("Fitted alpha:", alpha_hat)
print("Mean (alpha/sum):", alpha_hat / np.sum(alpha_hat))









# %%
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

def dirichlet_log_likelihood(alpha, X):
    """
    Compute the negative log-likelihood of the data X given alpha.
    We return the negative because we will use a minimizer.
    """
    alpha = np.array(alpha)
    # Sum of alphas
    alpha_0 = np.sum(alpha)
    
    # Log-likelihood for the entire dataset
    # Term 1: N * [ln Γ(α0) - Σ ln Γ(αk)]
    term1 = len(X) * (gammaln(alpha_0) - np.sum(gammaln(alpha)))
    
    # Term 2: Σ_i Σ_k (αk - 1) * ln(x_ik)
    term2 = np.sum((alpha - 1) * np.sum(np.log(X), axis=0))
    
    log_likelihood = term1 + term2
    return -log_likelihood # Return negative for minimization

# Your prepared data (N x 4)
# X = ... # Load and preprocess your CS_PERCENT data here

# Initial guess for alpha. A common, robust starting point is the
# method-of-moments estimator: alpha_k = (mean_k * (1 - mean_k) / var_k) - 1
# But a simple uniform start often works fine.
initial_alpha = np.array([1.0, 1.0, 1.0, 1.0])

# Bounds: Alpha values must be > 0.
bounds = [(1e-6, None) for _ in range(4)]

# Perform the optimization
result = minimize(
    fun=dirichlet_log_likelihood,
    x0=initial_alpha,
    args=(X,),
    method='L-BFGS-B', # A good choice for bounded problems
    bounds=bounds,
    options={'disp': True}
)

if result.success:
    estimated_alpha = result.x
    print("Estimated Alpha:", estimated_alpha)
else:
    print("Optimization failed:", result.message)
    # Fallback: Use method of moments or a default
    estimated_alpha = np.mean(X, axis=0) * 10 # A simple, arbitrary fallback




# %%
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, psi  # psi = digamma

def prepare_prob_rows(X, eps=1e-8):
    """Ensure each row is strictly positive and sums to 1."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array (N, K).")
    X = np.clip(X, eps, None) # avoid zeros
    X /= X.sum(axis=1, keepdims=True) # renormalize each row for rows whose CS percentages don’t sum to 1
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
    if np.any(alpha <= 0):
        return np.inf, np.full_like(alpha, np.nan)

    a0 = alpha.sum()
    # log-likelihood
    ll = N * (gammaln(a0) - np.sum(gammaln(alpha))) + np.dot(alpha - 1.0, S)
    # gradient of log-likelihood
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

    alpha = mu * alpha0   # This gives initial alpha for the whole dataset and for each condition state.
    # Ensure strictly positive alphas, even if some mu are 0/1
    alpha = np.clip(alpha, 1e-3, None)
    return alpha



def fit_dirichlet_mle_lbfgsb(X, x0=None, tol=1e-8, maxiter=1000, verbose=True):
    """
    Fit Dirichlet(alpha) by MLE using L-BFGS-B with analytic gradient.
    """
    X = prepare_prob_rows(X) # ensure valid input
    N, K = X.shape

    # sufficient statistics
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

file_path = 'G:/My Drive/PSU/Projects/PhD/WeeklyMeetings_Tasks/22_12102024_Soft_decision_tree/actCrit-vs-DynPrg-RL/AlphaForDrichlet/NBEExport_September_18_2025_12_51_19.txt'

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