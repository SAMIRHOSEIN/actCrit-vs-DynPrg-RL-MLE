# %%
# Final version to find the best match for alpha
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln,psi # gammaln = log gamma function, psi = digamma (derivative of log gamma function)
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize
from matplotlib import cm


# inputs
N_samples = 1000  # large so we see enough of the rare last state

# p_hat from the code (empirical mean from X_real)
# - n controls how “smooth” or “noisy” your synthetic samples are.
#       - If n is small--------------->    n=1     ---------------> one-hot vector
#       - If n is moderate-------------->  n=100  -------------->   smooth percentages, similar to real data
#       - If n is huge --------------- >   n=1000  -------------->  almost exactly equal to p_hat (too smooth)(too smooth, no variability)
# Also, The real dataset expresses condition states as proportions with two‐decimal accuracy(for example: 85 is 0.85), so n=100 is a good match.
N_cells = 100


file_path = './Datainfobridge/NBEExport_September_18_2025_12_51_19.txt'
cols = ["CS_PERCENT_1", "CS_PERCENT_2", "CS_PERCENT_3", "CS_PERCENT_4"] # the name of columns in the data file

rng = np.random.default_rng(43)  # reproducible random numbers


# Try to read as CSV (if it fails, try with sep="\t" for tab-delimited)
try:
    df = pd.read_csv(file_path)
except Exception:
    df = pd.read_csv(file_path, sep="\t")


# clean NaNs immediately
df = df.dropna(subset=cols).copy()
print(f"Number of bridges after dropping missing CS percentages: {len(df)}")

X = df[cols].values  # use this X for BOTH Dirichlet and Multinomial





########################################################
# 1) Estimate Dirichlet parameters by MLE with L-BFGS-B
########################################################
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

    
X = df[cols].values


alpha_hat, res = fit_dirichlet_mle_lbfgsb(X)
print("Estimated alpha:", alpha_hat)
print("Mean (alpha/sum):", alpha_hat / alpha_hat.sum())

# %%
#########################################################################
# 2) Estimate multinomial parameters 
#########################################################################

# ---------- 1) Utilities ----------
def fit_multinomial_from_cs(X):
    """
    Estimate a global multinomial parameter vector p from CS_PERCENT columns.

    Parameters
    ----------
    X : array-like, shape (N, K)
        Each row contains CS percentages or fractions for one bridge.

    Returns
    -------
    p_hat : (K,) array
        Estimated probability of each condition state.
    """
    X = prepare_prob_rows(X)       # convert percentages to probabilities
    p_hat = X.mean(axis=0)         # average across bridges
    p_hat /= p_hat.sum()           # just to be extra safe
    return p_hat


# ---------- 2) Load the real data and fit multinomial ----------
X = df[cols].values   # shape (N, 4)

p_hat = fit_multinomial_from_cs(X)
print("Estimated multinomial parameters p_hat:", p_hat)
print("Check sum(p_hat):", p_hat.sum())

# %%
#################################################################
# 3) Compare real data vs Dirichlet vs Multinomial (mean / std / plots)
#################################################################


# 1) Prepare real data as probability rows
X_real = prepare_prob_rows(X)                 # shape (N_bridges, 4)
mean_real = X_real.mean(axis=0)
std_real  = X_real.std(axis=0, ddof=1) # delta degrees of freedom = used to get denominator N-ddof for sample std dev

print("=== Real data (CS percentages) ===")
print("Mean per CS     :", mean_real)
print("Std dev per CS  :", std_real)
print()

# 2) Sample from Dirichlet(alpha_hat)
K = X_real.shape[1]
dir_samples = rng.dirichlet(alpha_hat, size=N_samples)   # shape (N_samples, 4)

mean_dir = dir_samples.mean(axis=0)
std_dir  = dir_samples.std(axis=0, ddof=1)

print("=== Dirichlet model ===")
print("alpha_hat        :", alpha_hat)
print("Mean per CS      :", mean_dir)
print("Std dev per CS   :", std_dir)
print()







# 3) Sample from Multinomial(p_hat) 
counts = rng.multinomial(N_cells, p_hat, size=N_samples)  # shape (N_samples, 4)
multi_frac = counts / N_cells                              # now rows sum to 1

# Use multi_frac instead of multi_samples in the rest of the analysis
multi_samples = multi_frac





mean_multi = multi_samples.mean(axis=0)
std_multi  = multi_samples.std(axis=0, ddof=1)

print(f"=== Multinomial composition model (N_cells = {N_cells}) ===")
print("p_hat            :", p_hat)
print("Mean per CS      :", mean_multi)
print("Std dev per CS   :", std_multi)
print()



# %%
#################################################################
# 4) Covariance comparison: Real vs Dirichlet vs Multinomial
#################################################################

def covariance_matrix(X):
    """
    Compute covariance matrix for probability vectors.
    X is (N, 4), each row = CS percentages or generated samples.
    Returns (4,4) covariance matrix.
    """
    return np.cov(X.T, ddof=1)   # transpose so shape = (4, N)


# Compute covariance matrices
cov_real = covariance_matrix(X_real)                 # (N_real, 4)
cov_dir  = covariance_matrix(dir_samples)            # (N_samples, 4)
cov_multi = covariance_matrix(multi_samples)         # (N_samples, 4)


#################################################################
# 5) Display in pandas tables for clean formatting
#################################################################
import pandas as pd

labels = ["CS1", "CS2", "CS3", "CS4"]

df_cov_real  = pd.DataFrame(cov_real,  index=labels, columns=labels)
df_cov_dir   = pd.DataFrame(cov_dir,   index=labels, columns=labels)
df_cov_multi = pd.DataFrame(cov_multi, index=labels, columns=labels)

print("======================================================")
print("Covariance Matrix — REAL DATA")
print("======================================================")
print(df_cov_real)
print()

print("======================================================")
print("Covariance Matrix — DIRICHLET")
print("======================================================")
print(df_cov_dir)
print()

print("======================================================")
print("Covariance Matrix — MULTINOMIAL")
print("======================================================")
print(df_cov_multi)
print()

#%%
###############################################################
# 6) Ternary contour plots for (CS1, CS2, CS3_new = CS3 + CS4)
# Figure 1: local scale (each plot its own z-range, 1×3)
# Figure 2: shared scale (same z-range for all, 1×3)
###############################################################
from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize
from matplotlib import cm

# ---------- 1) Compress 4 CSs -> 3 CSs (CS1, CS2, CS3_new) ----------

def compress_to_three(cs4_array):
    """
    cs4_array: (N, 4) with columns [CS1, CS2, CS3, CS4]
    Returns: (N, 3) with [CS1, CS2, CS3_new = CS3 + CS4], row-normalized.
    """
    cs1 = cs4_array[:, 0]
    cs2 = cs4_array[:, 1]
    cs3_new = cs4_array[:, 2] + cs4_array[:, 3]
    X3 = np.stack([cs1, cs2, cs3_new], axis=1)
    X3 = X3 / X3.sum(axis=1, keepdims=True)
    return X3

X3_real  = compress_to_three(X_real)
X3_dir   = compress_to_three(dir_samples)
X3_multi = compress_to_three(multi_samples)

sqrt3 = np.sqrt(3.0)

# ---------- 2) Barycentric -> 2D coordinates for ternary plot ----------

def ternary_to_cartesian(X3):
    """
    Map 3-component compositions (a,b,c), a+b+c=1,
    to 2D coordinates in an equilateral triangle.

    Vertices:
      CS1 -> (0, 0)
      CS2 -> (1, 0)
      CS3_new -> (0.5, sqrt(3)/2)
    """
    a = X3[:, 0]  # CS1
    b = X3[:, 1]  # CS2
    c = X3[:, 2]  # CS3_new
    x = b + 0.5 * c
    y = (sqrt3 / 2.0) * c
    return x, y

# ---------- 3) Compute KDE grid (xs, ys, z) for one dataset ----------

def compute_ternary_kde_grid(X3, n_grid=120):
    """
    X3: (N, 3), rows sum to 1
    Returns: xs, ys, z on a triangular grid.
    """
    X3 = X3 / X3.sum(axis=1, keepdims=True)
    x_data, y_data = ternary_to_cartesian(X3)
    data_xy = np.vstack([x_data, y_data])

    kde = gaussian_kde(data_xy)

    xs, ys = [], []
    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            k = n_grid - i - j
            a = i / n_grid
            b = j / n_grid
            c = k / n_grid
            xy = ternary_to_cartesian(np.array([[a, b, c]]))
            xs.append(xy[0][0])
            ys.append(xy[1][0])

    xs = np.array(xs)
    ys = np.array(ys)
    z = kde(np.vstack([xs, ys]))   # density on grid

    return xs, ys, z

# ---------- 4) Compute KDE grids for all three datasets ----------

xs_real,  ys_real,  z_real  = compute_ternary_kde_grid(X3_real,  n_grid=120)
xs_dir,   ys_dir,   z_dir   = compute_ternary_kde_grid(X3_dir,   n_grid=120)
xs_multi, ys_multi, z_multi = compute_ternary_kde_grid(X3_multi, n_grid=120)

# Pack them for easier looping
datasets = [
    ("Real Data (CS1, CS2, CS3_new)", xs_real,  ys_real,  z_real),
    ("Dirichlet Samples",             xs_dir,   ys_dir,   z_dir),
    (f"Multinomial Samples (N_cells = {N_cells})", xs_multi, ys_multi, z_multi),
]

# ---------- 5) Shared density bounds across all three ----------

z_all = np.concatenate([z_real, z_dir, z_multi])
z_min, z_max = z_all.min(), z_all.max()
norm = Normalize(vmin=z_min, vmax=z_max)
cmap = cm.viridis

print("Global density range:", z_min, z_max)

###############################################################
# FIGURE 1 — Local scale (each plot has its OWN colorbar)
###############################################################
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (title, xs, ys, z) in zip(axes, datasets):

    # Local normalization for *each* plot
    zmin_local, zmax_local = z.min(), z.max()
    norm_local = Normalize(vmin=zmin_local, vmax=zmax_local)
    levels_local = np.linspace(zmin_local, zmax_local, 20)

    sc = ax.tricontourf(xs, ys, z,
                        levels=levels_local,
                        cmap=cmap,
                        norm=norm_local)

    # Triangle border
    tri_x = [0.0, 1.0, 0.5, 0.0]
    tri_y = [0.0, 0.0, sqrt3/2, 0.0]
    ax.plot(tri_x, tri_y, "k-", lw=1.2)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=12)

    # Local colorbar for this subplot
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=norm_local, cmap=cmap),
        ax=ax,
        shrink=0.75
    )
    cbar.set_label("Local Density", fontsize=10)

fig.suptitle("Ternary KDE — LOCAL Density Scale", fontsize=16, y=1.05)
plt.tight_layout()
plt.show()


###############################################################
# FIGURE 2 — Shared scale (one colorbar for ALL)
###############################################################
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

levels_shared = np.linspace(z_min, z_max, 20)

for ax, (title, xs, ys, z) in zip(axes, datasets):

    sc = ax.tricontourf(xs, ys, z,
                        levels=levels_shared,
                        cmap=cmap,
                        norm=norm)

    # Triangle border
    tri_x = [0.0, 1.0, 0.5, 0.0]
    tri_y = [0.0, 0.0, sqrt3/2, 0.0]
    ax.plot(tri_x, tri_y, "k-", lw=1.2)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=12)

# Shared colorbar on the right
cbar = fig.colorbar(
    cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=axes,
    shrink=0.75,
    location="right"
)
cbar.set_label("Shared Density", fontsize=12)

fig.suptitle("Ternary KDE — SHARED Density Scale", fontsize=16, y=1.05)
plt.tight_layout()
plt.show()

# %%
#################################################################
# 7) 1D marginal distributions for each condition state
#    Real vs Dirichlet vs Multinomial — SEPARATE FIGURES
#################################################################

cs_labels = ["CS1", "CS2", "CS3", "CS4"]

N_plot = min(50_000, dir_samples.shape[0], multi_samples.shape[0])
idx_plot = rng.choice(dir_samples.shape[0], size=N_plot, replace=False)

dir_plot   = dir_samples[idx_plot]
multi_plot = multi_samples[idx_plot]

for k in range(K):

    plt.figure(figsize=(6, 5))
    plt.hist(
        X_real[:, k],
        bins=50, range=(0, 1),
        density=True, alpha=0.6,
        color="red",
        label="Real data"
    )

    plt.hist(
        dir_plot[:, k],
        bins=50, range=(0, 1),
        density=True, alpha=0.6,
        color="blue",
        label="Dirichlet"
    )

    plt.hist(
        multi_plot[:, k],
        bins=50, range=(0, 1),
        density=True, alpha=0.6,
        color="green",
        label=f"Multinomial (N_cells={N_cells})"
    )

    plt.title(f"Distribution of {cs_labels[k]}")
    plt.xlabel("Fraction in this condition state")
    plt.ylabel("Density")
    plt.xlim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()
# %%
