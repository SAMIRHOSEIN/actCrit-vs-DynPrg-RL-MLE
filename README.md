# Actor-Critic vs. Dynamic Programming for Bridge-Element Maintenance

> Life-cycle-cost-optimal maintenance planning for a deteriorating bridge
> element, framed as a Markov Decision Process and solved with **Dynamic
> Programming** (an exact baseline) and **PPO actor-critic Reinforcement
> Learning** — with **interpretable soft/oblique decision-tree policies** and a
> statistical front-end that fits the initial condition-state distribution from
> **real National Bridge Element (NBE) inspection data**.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-TorchRL-ee4c2c.svg">
  <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg">
  <a href="https://arxiv.org/abs/2604.02528"><img alt="Paper" src="https://img.shields.io/badge/paper-arXiv%3A2604.02528-b31b1b.svg"></a>
  <img alt="Status" src="https://img.shields.io/badge/status-research-orange.svg">
</p>

> 📄 **Companion code for:** S. A. Moayyedi and D. Y. Yang, *"Interpretable Deep
> Reinforcement Learning for Element-level Bridge Life-cycle Optimization,"*
> Portland State University, 2026 — **arXiv:2604.02528** (under review).
> See [Citation](#citation).

---

## Table of contents

- [Motivation](#motivation)
- [What this repository does](#what-this-repository-does)
- [Problem formulation](#problem-formulation-the-mdp)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [License](#license)

---

## Motivation

The 2022 **Specifications for the National Bridge Inventory (SNBI)** represent a
bridge element's condition as a **four-dimensional array of condition-state (CS)
proportions** rather than a single categorical rating. This granularity is
valuable, but it dramatically expands the state space over which owners must
decide **when and how** to intervene — *do nothing*, *maintain*, *repair*,
*rehabilitate*, or *replace* — to minimize the **risk-included life-cycle cost
(LCC)**: the discounted sum of intervention costs plus the expected cost of
failure.

This project treats that decision problem as a **Markov Decision Process (MDP)**.
It is the companion code for the paper above and focuses on **two concrete
goals**:

> **Goal 1 — Model the initial condition.** Fit **Dirichlet** and **Multinomial**
> models to real bridge-element inspection data, compare both against the data
> (in 1-D, 2-D, and 3-D), pick the better-fitting distribution, and use its
> parameters (the Dirichlet concentration vector **α**) as the environment's
> initial condition-state distribution.
>
> **Goal 2 — Compare DP with PPO.** Solve the maintenance MDP with **Dynamic
> Programming** (value iteration — the exact optimum) and evaluate it in the
> **same environment** used to train **PPO actor-critic RL**, so the two
> approaches are compared on identical dynamics, costs, and initial conditions.

The inspection data are **real FHWA InfoBridge records** for Oregon
state-highway steel-girder bridges; the distributions are fit via Maximum
Likelihood (L-BFGS-B with analytic gradients) and the Method of Moments.

## What this repository does

| Component | File(s) | Description |
|---|---|---|
| **Environment** | `bridge_gym/example_nbe107/` | A `gymnasium` MDP for a single steel girder/beam element (NBE 107): 4 condition states, 5 maintenance actions, cost-shaped reward. |
| **Dynamic Programming** | `DPvsPPO.py` | Finite-horizon and stationary value iteration; the exact optimal baseline. Includes a hand-verified toy example. |
| **PPO actor-critic** | `softtree_ppo/` | A TorchRL PPO trainer (`PPOTrainer`) plus a soft-tree variant with entropy/L1/L2/group-L1 regularization and β-annealing. |
| **Interpretable policies** | `softtree/` | Soft decision-tree classifier, pruning, and conversion to a compact **oblique decision tree** with text/Graphviz visualization. |
| **Data modeling (MLE/MoM)** | `Dir_MLE_MOM_MultiNomin.py` | Fits Dirichlet (MLE with analytic gradient, L-BFGS-B) and Multinomial models to NBE condition-state percentages; ternary/marginal diagnostics and KS goodness-of-fit. |
| **Convergence & stats** | `convergence.py`, `eval_stats.py` | Mean ± 95% CI of episode returns and a precision-based stopping rule for the number of evaluation episodes. |

## Problem formulation (the MDP)

- **State** — a probability distribution over 4 condition states
  `[CS1, CS2, CS3, CS4]` (best → worst), optionally augmented with normalized
  time. The initial state is sampled from a `Dirichlet(α)` fitted to real data.
- **Actions** — `{0: Do nothing, 1: Maintenance, 2: Repair, 3: Rehabilitation,
  4: Replacement}`, each with its own Markov transition matrix and unit cost.
- **Dynamics** — `s' = Pᵀ(a) · s`, renormalized for numerical stability.
- **Reward** — the negative of the per-step cost,

  ```
  cost(s, a) = unit_cost(a)·s  +  (p_fail·s)·failure_cost
  reward     = −cost                         (discounted by γ = 1/1.03)
  ```

- **Objective** — maximize expected discounted return ⇔ **minimize LCC**.

Deterioration and cost assumptions are documented inline in
`bridge_gym/example_nbe107/settings.py` (transition matrices adapted from
Thompson et al. 1998, CoRe element 107, migrated to the NBE convention per the
AASHTO Bridge Element Inspection Manual).

## Repository structure

```
actCrit-vs-DynPrg-RL-MLE/
├── DPvsPPO.py                     # DP value iteration + evaluation in the RL env
├── Dir_MLE_MOM_MultiNomin.py      # Dirichlet/Multinomial fitting to NBE data
├── convergence.py                 # Episode-count convergence analysis
├── eval_stats.py                  # Mean & confidence-interval helper
│
├── bridge_gym/                    # Gymnasium environment package
│   ├── debug_example_nbe107.py    # Manual rollout / env-spec sanity checks
│   └── example_nbe107/
│       ├── settings.py            # Transition matrices, costs, failure probs
│       ├── rl_env.py              # SingleElement gym.Env
│       └── cost_util.py           # Cost → reward transforms
│
├── softtree/                      # Interpretable decision-tree tooling
│   ├── softtree_classification.py # Differentiable soft decision tree
│   ├── training_util.py           # Supervised soft-tree training loop
│   ├── extraction_util.py         # Node pruning
│   └── oblique_tree.py            # Soft tree → compact oblique tree (+viz)
│
├── softtree_ppo/                  # PPO actor-critic training
│   ├── training.py                # PPOTrainer / SofttreePPOTrainer
│   ├── rl_util.py                 # Actor & critic networks
│   └── settings.py
│
├── V1_.. V4_Datainfobridge*/      # NBE inspection exports (input data)
├── plot/                          # Generated figures
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Installation

Requires **Python 3.10+**.

```bash
git clone https://github.com/SAMIRHOSEIN/actCrit-vs-DynPrg-RL-MLE.git
cd actCrit-vs-DynPrg-RL-MLE

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> **TorchRL note.** `torchrl` and `tensordict` are tightly version-coupled;
> install matching releases. The code targets a recent TorchRL (≈0.11) — see the
> comments in `softtree_ppo/training.py` for the older 0.9.2 API equivalents.
>
> **Graphviz note.** Rendering oblique-tree diagrams needs the Graphviz *system*
> package in addition to the `graphviz` Python binding
> (`apt install graphviz` / `brew install graphviz`).

## Usage

**1 — Fit the initial condition-state distribution from inspection data**

```bash
python Dir_MLE_MOM_MultiNomin.py
```
Estimates `Dirichlet(α)` and Multinomial parameters, runs a synthetic
parameter-recovery sanity check, and writes marginal-fit figures to `plot/`.
The fitted `α` feeds the environment's `dirichlet_alpha` reset distribution.

**2 — Solve and evaluate the Dynamic-Programming policy**

```bash
python DPvsPPO.py
```
Runs value iteration (stationary by default; set `ELE_DP_INC_STEP = True` for
finite-horizon), evaluates the DP policy in the environment over many episodes,
and reports mean episode return with a 95% confidence interval.

**3 — Assess evaluation-count convergence**

```bash
python convergence.py
```
Parses the printed `mean / 95% CI / SD / N` summary lines and reports the
smallest N whose CI half-width falls within the relative tolerance.

**4 — Sanity-check the environment**

```bash
python -m bridge_gym.debug_example_nbe107
```
Manual rollout plus a TorchRL `check_env_specs` integrity test.

> The PPO trainer lives in `softtree_ppo/training.py` (`PPOTrainer` /
> `SofttreePPOTrainer`) and is driven programmatically; see
> [Roadmap](#roadmap).

## Methodology

**Dynamic Programming (exact baseline).** Both finite-horizon backward induction
and infinite-horizon discounted value iteration are implemented from first
principles for transparency, and validated against a 2-state / 2-action example
with hand-computed values.

**PPO actor-critic (learned policy).** Built on TorchRL with GAE, clipped
surrogate loss, optional entropy bonus, and L1/L2/group-L1 actor regularization.
A soft-tree variant additionally anneals the routing temperature (β) during
training so the policy sharpens toward hard, interpretable splits.

**Interpretable policies.** The PPO actor can be a differentiable soft decision
tree, which is pruned and converted into a compact **oblique decision tree**
whose splits are linear rules on the condition-state vector — small enough to
read and audit.

**Data modeling (MLE / MoM).** The Dirichlet fit uses L-BFGS-B with an analytic
gradient and a method-of-moments initialization (Minka / Ronning); fit quality
is checked against the Multinomial model with ternary KDE plots (2-D and 3-D),
per-CS marginals, and KS goodness-of-fit.

## Results

All figures below are produced directly by the scripts in this repository.

### Goal 1 — Which distribution fits the real data?

`Dir_MLE_MOM_MultiNomin.py` fits a Dirichlet and a Multinomial model to the real
inspection data and compares them against the data. Two views make the
difference clear (reducing the four condition states to three,
`CS3_new = CS3 + CS4`, lets the compositions be drawn on the probability
simplex):

**3-D density on the simplex — real vs. Dirichlet vs. Multinomial.** The real
data piles up near the CS1 corner and along the CS1–CS2 edge. The Dirichlet
reproduces this boundary-concentrated shape; the Multinomial collapses to a
single narrow spike.

<p align="left">
  <img src="plot/ternary_kde_3d_comparison.png" width="100%">
</p>

**1-D marginal for CS1 — real (red) vs. Dirichlet (blue) vs. Multinomial
(green).** The real marginal is U-shaped (mass at 0 and 1); the Dirichlet tracks
it, while the Multinomial forms an incorrect central bump.

<p align="left">
  <img src="plot/marginal_compare_CS1.png" width="55%">
</p>

**Why the Dirichlet is chosen.** A bridge element's condition is a vector of four
condition-state *proportions that sum to one* — a composition on the probability
simplex — and the Dirichlet is the natural distribution over such compositions.
The Multinomial instead models integer counts and, for a fixed number of cells,
is far too concentrated to represent the true diversity of bridge conditions.
This diversity matters for training: a policy trained only from a brand-new
element (`s = [1, 0, 0, 0]`) leaves most of the state space unexplored, so each
training episode is instead restarted from a Dirichlet-sampled condition, which
exposes the policy to the full range of realistic (and worse-than-average)
conditions. The fitted concentration vector **α̂** is therefore adopted as the
environment's initial condition-state distribution.

The Dirichlet-implied Beta marginal for each condition state (orange) against
the real data (blue):

<p align="left">
  <img src="plot/beta_marginal_CS1.png" width="24%">
  <img src="plot/beta_marginal_CS2.png" width="24%">
  <img src="plot/beta_marginal_CS3.png" width="24%">
  <img src="plot/beta_marginal_CS4.png" width="24%">
</p>

### Goal 2 — Dynamic Programming vs. PPO in the same environment

`DPvsPPO.py` solves the MDP with value iteration and evaluates the resulting
policy in the **same** `SingleElement` environment that PPO trains and is
evaluated in, so both methods see identical dynamics, costs, and (Dirichlet-
sampled) initial conditions. Below, each point is one of **1,000 validation
episodes**: as the initial reliability index **β** increases (a healthier
starting element), the episode **life-cycle cost** drops sharply — the DP policy
spends less on interventions when the element starts in good condition.

<p align="left">
  <img src="plot/dp_initial_beta_vs_LCC.png" width="60%">
</p>

`DPvsPPO.py` also reports the mean episode return with a 95% confidence
interval, the action distribution, and a color-coded action timeline;
`convergence.py` shows how the return estimate tightens as the evaluation-episode
count grows. The head-to-head DP-vs-PPO comparison under this shared environment
is presented in the accompanying paper.

## Roadmap

- [ ] Add a runnable PPO training entry-point script (train → evaluate →
      compare against the DP baseline in one command).
- [ ] Publish a consolidated DP-vs-PPO results table and figures.
- [ ] Package the modules (`pyproject.toml`) for `pip install -e .`.
- [ ] Add unit tests for the environment dynamics and value-iteration correctness.

## Citation

If you use this code or its results, please cite the accompanying paper (see
also [`CITATION.cff`](CITATION.cff)):

> S. A. Moayyedi and D. Y. Yang, "Interpretable Deep Reinforcement Learning for
> Element-level Bridge Life-cycle Optimization," Portland State University, 2026.
> arXiv:2604.02528 (under review).

```bibtex
@article{moayyedi2026interpretable,
  title   = {Interpretable Deep Reinforcement Learning for Element-level Bridge Life-cycle Optimization},
  author  = {Moayyedi, Seyyed Amirhossein and Yang, David Y.},
  journal = {arXiv preprint arXiv:2604.02528},
  year    = {2026},
  note    = {Under review}
}
```

## License

Released under the [MIT License](LICENSE). © 2025 Amir Moayyedi.
