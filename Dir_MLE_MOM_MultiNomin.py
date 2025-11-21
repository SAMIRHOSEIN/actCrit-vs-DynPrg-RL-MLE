# %%
# imports
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln,psi # gammaln = log gamma function, psi = digamma (derivative of log gamma function)
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize
from matplotlib import cm
########################################################
# 0) inputs
########################################################
N_samples = 1000  # large so we see enough of the rare last state

# p_hat from the code (empirical mean from X_real)
# - n controls how “smooth” or “noisy” your synthetic samples are.
#       - If n is small--------------->    n=1     ---------------> one-hot vector
#       - If n is moderate-------------->  n=100  -------------->   smooth percentages, similar to real data
#       - If n is huge --------------- >   n=1000  --------------->  almost exactly equal to p_hat (too smooth)(too smooth, no variability)
# Also, The real dataset expresses condition states as proportions with two‐decimal accuracy(for example: 85 is 0.85), so n=100 is a good match.
# In addition, I chose N_cells = 100 because one bridge deck can be interpreted as 100 virtual units, allowing me to approximate percentages with reasonable variability while keeping the variance consistent with real data.
N_cells = 100


file_path = './Datainfobridge/NBEExport_September_18_2025_12_51_19.txt'
cols = ["CS_PERCENT_1", "CS_PERCENT_2", "CS_PERCENT_3", "CS_PERCENT_4"] # the name of columns in the data file
df = pd.read_csv(file_path)

# labels for covariance matrix and plots
cs_labels = ["CS1", "CS2", "CS3", "CS4"]

rng = np.random.default_rng(42)  # reproducible random numbers

# clean NaNs
df = df.dropna(subset=cols).copy()
print(f"Number of bridges after dropping missing CS percentages: {len(df)}")

X = df[cols].values  # use this X for BOTH Dirichlet and Multinomial



# Inputs for sanity check of dirichlet and multinomial
alpha_true = np.array([2.0, 5.0, 3.0, 1.0])  # some asymmetric Dirichlet
N_bridges = 5000   # number of synthetic "bridges"

########################################################
# 1) Estimate Dirichlet parameters by MLE with L-BFGS-B
########################################################
def prepare_prob_rows(X, eps=1e-8):
    """Ensure each row is strictly positive and sums to 1."""
    X = np.asarray(X, dtype=float)
    X = np.clip(X, eps, None)           # Ensures all entries are at least eps to avoid zeros due to log(0)
    X /= X.sum(axis=1, keepdims=True)   # Renormalize each row for rows whose CS percentages don’t sum to 1 (sum of rows in original data may not be exactly 1)
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
    # why we defined gradient of log-likelihood and why do we need this?
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
    X = prepare_prob_rows(X)       
    p_hat = X.mean(axis=0)         
    return p_hat

# Load the real data and fit multinomial
X = df[cols].values   # shape (N, 4)
p_hat = fit_multinomial_from_cs(X)
print("Estimated multinomial parameters p_hat:", p_hat)
print("Check sum(p_hat):", p_hat.sum())

# %%
###########################################################
# 3) SANITY CHECK: Dirichlet + Multinomial on synthetic data
###########################################################
def sanity_check_dirichlet_multinomial(alpha_true, N_bridges_test=5000, N_cells=100):
    """
    1) Pick a known Dirichlet parameter alpha_true.
    2) Sample many compositions from Dirichlet(alpha_true).
       - Fit Dirichlet with my MLE code and check that alpha_hat ≈ alpha_true.
    3) Use p_true = alpha_true / sum(alpha_true) as the "true" multinomial p.
        - Note: I consider p_true = alpha_true / sum(alpha_true) because in Dirichlet, the mean of each component is given by alpha_k / sum(alpha).
       - Sample multinomial counts, convert to percentages.
       - Fit multinomial with my code and check that p_hat ≈ p_true.
    4) For Dirichlet, compare theoretical vs empirical variance/covariance.
    """

    # 1) Define ground-truth parameters
    alpha0_true = alpha_true.sum()
    p_true = alpha_true / alpha0_true           # mean of Dirichlet(the mean of each component is given by alpha_k / sum(alpha)) for multinominal check

    # 2) Generate Dirichlet synthetic data 
    X_dir_synth = rng.dirichlet(alpha_true, size=N_bridges_test)

    # Fit Dirichlet using my MLE code
    alpha_hat_test, _ = fit_dirichlet_mle_lbfgsb(X_dir_synth, verbose=False)

    print("==============================================")
    print("DIRICHLET SANITY CHECK (parameter recovery)")
    print("==============================================")
    print("alpha_true :", alpha_true)
    print("alpha_hat  :", alpha_hat_test)
    print("||alpha_hat - alpha_true||_2 =",
          np.linalg.norm(alpha_hat_test - alpha_true))

    # 3) Generate Multinomial synthetic data 
    counts = rng.multinomial(N_cells, p_true, size=N_bridges_test)
    X_multi_synth = counts / N_cells  # convert to fractions

    p_hat_multi = fit_multinomial_from_cs(X_multi_synth)

    print("\n==============================================")
    print("MULTINOMIAL SANITY CHECK (parameter recovery)")
    print("==============================================")
    print("p_true      :", p_true)
    print("p_hat_multi :", p_hat_multi)
    print("||p_hat_multi - p_true||_2 =",
          np.linalg.norm(p_hat_multi - p_true))

# Call this once to run the sanity check:
sanity_check_dirichlet_multinomial(alpha_true, N_bridges_test=N_bridges, N_cells=N_cells)
#%%
#################################################################
# 4) Compare real data vs Dirichlet vs Multinomial (mean / std / plots)
#################################################################
# preparedness
X_real = prepare_prob_rows(X)                 # shape (N_bridges, 4)

# 1-real data
mean_real = X_real.mean(axis=0)
std_real  = X_real.std(axis=0, ddof=1) 
print("\n==============================================")
print("Compare Real Data vs Dirichlet vs Multinomial")
print("==============================================")
print("=== Real data (CS percentages) ===")
print("Mean per CS     :", mean_real)
print("Std dev per CS  :", std_real)
print()

# 2-Sampling and computing mean and std for Dirichlet(alpha_hat)
dir_samples = rng.dirichlet(alpha_hat, size=N_samples)  
mean_dir = dir_samples.mean(axis=0)
std_dir  = dir_samples.std(axis=0, ddof=1)
print("=== Dirichlet model ===")
print("alpha_hat        :", alpha_hat)
print("Mean per CS      :", mean_dir)
print("Std dev per CS   :", std_dir)
print()

# 3- Sampling and computing mean and std for Multinomial(p_hat) 
counts = rng.multinomial(N_cells, p_hat, size=N_samples)  # shape (N_samples, 4)
multi_samples = counts / N_cells                           # now rows sum to 1
mean_multi = multi_samples.mean(axis=0)
std_multi  = multi_samples.std(axis=0, ddof=1)
print(f"=== Multinomial composition model (N_cells = {N_cells}) ===")
print("p_hat            :", p_hat)
print("Mean per CS      :", mean_multi)
print("Std dev per CS   :", std_multi)
print()
# %%
#################################################################
# 5) Covariance comparison: Real vs Dirichlet vs Multinomial
#################################################################
def covariance_matrix(X):
    """
    Compute covariance matrix for probability vectors.
    X is (N, 4), each row = CS percentages or generated samples.
    Returns (4,4) covariance matrix.
    """
    return np.cov(X.T, ddof=1)   # transpose so shape = (4, N)

cov_real = covariance_matrix(X_real)                 # (N_real, 4)
cov_dir  = covariance_matrix(dir_samples)            # (N_samples, 4)
cov_multi = covariance_matrix(multi_samples)         # (N_samples, 4)
#################################################################
# 5) Display in pandas tables for clean formatting
#################################################################

df_cov_real  = pd.DataFrame(cov_real,  index=cs_labels, columns=cs_labels)
df_cov_dir   = pd.DataFrame(cov_dir,   index=cs_labels, columns=cs_labels)
df_cov_multi = pd.DataFrame(cov_multi, index=cs_labels, columns=cs_labels)

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
# Figure 1: scatter-only plots
# Figure 2: local scale (each plot its own z-range, 1×3)
# Figure 3: Same scale (same z-range for all, 1×3)
###############################################################
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

# Barycentric to 2D coordinates(Cartesian) for ternary plot 
def ternary_to_cartesian(X3):
    """
    Map 3-component compositions (a,b,c), a+b+c=1,
    to 2D coordinates in an equilateral triangle.

    [CS1  CS2  CS3_new] = (a, b, c) -----> (x,y):
        CS1: (1,0,0) -> (0,0)
        CS2: (0,1,0) -> (1,0)
        CS3_new: (0,0,1) -> (0.5, sqrt(3)/2)
    """
    a = X3[:, 0]  # CS1
    b = X3[:, 1]  # CS2
    c = X3[:, 2]  # CS3_new
    x = b + 0.5 * c
    y = (sqrt3 / 2.0) * c
    return x, y

# Compute KDE grid (xs, ys, z) for one dataset
def compute_ternary_kde_grid(X3, n_grid=120):
    """
    X3: (N, 3), rows sum to 1
    Returns: xs, ys, z on a triangular grid.

    """
    X3 = X3 / X3.sum(axis=1, keepdims=True)
    x_data, y_data = ternary_to_cartesian(X3)
    data_xy = np.vstack([x_data, y_data])

    kde = gaussian_kde(data_xy) # probability density estimator

    xs, ys = [], []
    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            k = n_grid - i - j
            # Generates all points inside the triangle using barycentric coordinates
            a = i / n_grid
            b = j / n_grid
            c = k / n_grid
            xy = ternary_to_cartesian(np.array([[a, b, c]]))
            xs.append(xy[0][0])
            ys.append(xy[1][0])

    xs = np.array(xs)
    ys = np.array(ys)
    z = kde(np.vstack([xs, ys]))   # density on grid

    return xs, ys, z, x_data, y_data

# Compute KDE grids for all three datasets
xs_real,  ys_real,  z_real,  x_real,  y_real = compute_ternary_kde_grid(X3_real,  n_grid=120)
xs_dir,   ys_dir,   z_dir,   x_dir,   y_dir   = compute_ternary_kde_grid(X3_dir,   n_grid=120)
xs_multi, ys_multi, z_multi, x_multi, y_multi = compute_ternary_kde_grid(X3_multi, n_grid=120)

# Pack them for easier looping
datasets = [
    ("Real Data (CS1, CS2, CS3_new)", xs_real, ys_real, z_real, x_real, y_real),
    ("Dirichlet Samples", xs_dir,  ys_dir,  z_dir,  x_dir,  y_dir),
    (f"Multinomial Samples (N_cells = {N_cells})", xs_multi, ys_multi, z_multi, x_multi, y_multi),
]

# compute z_min and z_max from *only these datasets cause sometines I want to compare two of three distributionns
z_min = min(z.min() for (_, _, _, z, _, _) in datasets)
z_max = max(z.max() for (_, _, _, z, _, _) in datasets)
norm  = Normalize(vmin=z_min, vmax=z_max)
# print("Same scale recomputed from selected datasets:")
# print("z_min:", z_min, "   z_max:", z_max)

# Same density bounds across all three 
cmap = cm.viridis

# ==========================================
# FIGURE 1 — Scatter-only ternary plots
# ==========================================
fig_scatter, axes_scatter = plt.subplots(1, 3, figsize=(18, 5), facecolor="white")

for ax, (title, xs, ys, z, x_data, y_data) in zip(axes_scatter, datasets):
    # white background for each axis
    ax.set_facecolor("white")

    # dashed triangle border
    tri_x = [0.0, 1.0, 0.5, 0.0]
    tri_y = [0.0, 0.0, sqrt3/2, 0.0]
    ax.plot(tri_x, tri_y, linestyle="--", color="black", lw=1.2)

    # scatter points (use a visible color, small crosses)
    ax.scatter(
        x_data,
        y_data,
        s=6,
        color="navy",
        alpha=0.6,
        marker="x",
        linewidths=0.4,
    )

    # labels at vertices
    ax.text(-0.04, -0.04, "CS1", ha="right", va="top", fontsize=10)
    ax.text(1.04, -0.04, "CS2", ha="left",  va="top", fontsize=10)
    ax.text(0.5,  sqrt3/2 + 0.06, "CS3_new",
            ha="center", va="bottom", fontsize=10)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=12)

fig_scatter.suptitle("Ternary KDE — Scatter", fontsize=16, y=1.05)
# plt.tight_layout()
plt.show()


###############################################################
# FIGURE 2 — Local scale (each plot has its OWN colorbar)
###############################################################
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (title, xs, ys, z, _, _) in zip(axes, datasets):

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

    ax.text(-0.04, -0.04, "CS1", ha="right", va="top", fontsize=10)
    ax.text(1.04, -0.04, "CS2", ha="left",  va="top", fontsize=10)
    ax.text(0.5,  sqrt3/2 + 0.06, "CS3_new",
            ha="center", va="bottom", fontsize=10)

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
# plt.tight_layout()
plt.show()

###############################################################
# FIGURE 3 — Same scale (one colorbar for ALL)
###############################################################
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

levels_shared = np.linspace(z_min, z_max, 20)

for ax, (title, xs, ys, z, _, _) in zip(axes, datasets):

    sc = ax.tricontourf(xs, ys, z,
                        levels=levels_shared,
                        cmap=cmap,
                        norm=norm)

    # Triangle border
    tri_x = [0.0, 1.0, 0.5, 0.0]
    tri_y = [0.0, 0.0, sqrt3/2, 0.0]
    ax.plot(tri_x, tri_y, "k-", lw=1.2)

    ax.text(-0.04, -0.04, "CS1", ha="right", va="top", fontsize=10)
    ax.text(1.04, -0.04, "CS2", ha="left",  va="top", fontsize=10)
    ax.text(0.5,  sqrt3/2 + 0.06, "CS3_new",
            ha="center", va="bottom", fontsize=10)

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

fig.suptitle("Ternary KDE — Same Density Scale", fontsize=16, y=1.05)
# plt.tight_layout()
plt.show()

# %%
#################################################################
# 7) 1D marginal distributions for each condition state
#    Real vs Dirichlet vs Multinomial — SEPARATE FIGURES
#################################################################
N_plot = min(50_000, dir_samples.shape[0], multi_samples.shape[0])
idx_plot = rng.choice(dir_samples.shape[0], size=N_plot, replace=False)

dir_plot   = dir_samples[idx_plot]
multi_plot = multi_samples[idx_plot]

K = X_real.shape[1]

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
    # plt.tight_layout()
    plt.show()
# %%
#################################################################
# 8) 3D ternary KDE surfaces (LOCAL density scale)
#################################################################
fig_local = plt.figure(figsize=(18, 6))
axes_3d_local = []

for i, (title, xs, ys, z, _, _) in enumerate(datasets):
    ax = fig_local.add_subplot(1, 3, i + 1, projection="3d")
    axes_3d_local.append(ax)

    # local norm and levels for THIS dataset only
    zmin_local, zmax_local = z.min(), z.max()
    norm_local = Normalize(vmin=zmin_local, vmax=zmax_local)

    surf = ax.plot_trisurf(
        xs,
        ys,
        z,
        cmap=cmap,
        norm=norm_local,     # <-- LOCAL scale here
        linewidth=0.0,
        antialiased=True,
    )

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Density")

    # triangle vertices in x–y plane
    v1 = (0.0, 0.0)          # CS1
    v2 = (1.0, 0.0)          # CS2
    v3 = (0.5, np.sqrt(3)/2) # CS3_new
    z_label = zmin_local - 0.02 * (zmax_local - zmin_local)

    ax.text(v1[0], v1[1], z_label, "CS1",
            ha="center", va="top", fontsize=11)
    ax.text(v2[0], v2[1], z_label, "CS2",
            ha="center", va="top", fontsize=11)
    ax.text(v3[0], v3[1], z_label, "CS3_new",
            ha="center", va="bottom", fontsize=11)

    ax.set_zlim(zmin_local, zmax_local)

    # local colorbar for THIS subplot
    cbar_local = fig_local.colorbar(
        cm.ScalarMappable(norm=norm_local, cmap=cmap),
        ax=ax,
        shrink=0.75,
        location="right",
    )
    cbar_local.set_label("Local Density", fontsize=10)

# plt.tight_layout()
plt.show()

#################################################################
# 9) 3D ternary KDE surfaces (shared density scale)
#################################################################

fig = plt.figure(figsize=(18, 6))
axes_3d = []

levels_shared = np.linspace(z_min, z_max, 20)  # same as 2D shared plot

for i, (title, xs, ys, z, _, _) in enumerate(datasets):
    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    axes_3d.append(ax)

    # 3D surface over the triangular grid
    surf = ax.plot_trisurf(
        xs,
        ys,
        z,
        cmap=cmap,
        norm=norm,          # same density scale for all three
        linewidth=0.0,
        antialiased=True,
    )

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Density")


    # triangle vertices in x–y plane
    v1 = (0.0, 0.0)          # CS1
    v2 = (1.0, 0.0)          # CS2
    v3 = (0.5, np.sqrt(3)/2) # CS3_new

    # place labels slightly above the triangle (z = z_min)
    z_label = z_min - 0.02*(z_max - z_min)

    ax.text(v1[0], v1[1], z_label, "CS1",
            ha="center", va="top", fontsize=11)

    ax.text(v2[0], v2[1], z_label, "CS2",
            ha="center", va="top", fontsize=11)

    ax.text(v3[0], v3[1], z_label, "CS3_new",
            ha="center", va="bottom", fontsize=11)


    # Same z-limits for comparability
    ax.set_zlim(z_min, z_max)

# One shared colorbar for all three 3D plots
cbar = fig.colorbar(
    cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=axes_3d,
    shrink=0.75,
    location="right"
)
cbar.set_label("Same Density Scale", fontsize=12)

# fig.suptitle("Ternary KDE — 3D Surfaces with Same Density Scale", fontsize=16, y=0.95)
# plt.tight_layout()
plt.show()
# %%
