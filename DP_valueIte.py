# %%
# 0) Imports & reproducibility
# DP vs PPO — **Aligned Inputs**
# This notebook aligns Dynamic Programming (finite-horizon value iteration) with our PPO environment so both share:
# - the same state/action spaces (ncs, na),
# - the same transition matrices (action_model),
# - the same reward/cost definition (unit_costs, cs_pfs, failure_cost),
# - the same discount factor gamma and horizon.

# Source of truth: element/problem_setup.py and element/rl_env.py.

import numpy as np
import matplotlib.pyplot as plt

from element.problem_setup import (
    ncs, na, gamma, action_model, unit_costs, failure_cost, cs_pfs
)

from test_constants import ELE_PPO_MAX_COST

from element.utility_func import cost_util



# %%
## 1) Build DP model directly from the PPO env parameters
# We construct:
# - P with shape (A, S, S) using action_model[a] (rows: next state, cols: current state), consistent with `env` where next = A[a].T @ state` ⇒ column-wise transitions.
# - R with shape (A, S) using the same cost definition used in the env step:
#   #   cost = unit_costs + failure_cost * cs_pfs
#   and reward via `cost_util(cost, min_cost=0, max_cost=unit_costs.max())`.
# 1- Build P and R from env constants
A, S = na, ncs 

# Transition tensors (A, S, S): P[a, s, s'] = Prob(s' | s, a)
# NOTE: action_model[a] is SxS with columns=current state, rows=next state,
# so P[a] := action_model[a]
P = np.array([action_model[a] for a in range(A)])  # (A, S, S)

# Rewards (A, S): immediate reward at pure state s under action a
ma_cost = ELE_PPO_MAX_COST
R = np.zeros((A, S), dtype=np.float64)

for a in range(A):
    for s in range(S):
        direct_cost = unit_costs[a,s]
        # print(f"Action {a}, State {s}, Direct cost: {direct_cost}")
        fail_risk = cs_pfs[s] * failure_cost
        cost = direct_cost + fail_risk
        R[a,s] = cost_util(cost, min_cost=0, max_cost=ma_cost)
# %%
