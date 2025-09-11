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
from element.problem_setup import (ncs, na, gamma, action_model, unit_costs, failure_cost, cs_pfs)
from test_constants import (ELE_DP_MAX_COST, ELE_DP_HORIZON)
from element.utility_func import cost_util
# %%
#--------------------------------------------------------------------------------------------------------------------------------------------------------
# 1) Build DP model directly from the PPO env parameters
# We construct:
# - P with shape (A, S, S) using action_model[a] (rows: next state, cols: current state), consistent with `env` where next = A[a].T @ state` ⇒ column-wise transitions.
# - R with shape (A, S) using the same cost definition used in the env step:
#   # cost = unit_costs + failure_cost * cs_pfs
#   and reward via `cost_util(cost, min_cost=0, max_cost=unit_costs.max())`.
#Build P and R from env constants
A, S = na, ncs 

# Note: action_model[a][s', s] = P(s'|s,a)
P = np.array([action_model[a] for a in range(A)])  # (A, S, S)

# Rewards (A, S): immediate reward at pure state s under action a
ma_cost = ELE_DP_MAX_COST
R = np.zeros((A, S), dtype=np.float64)

for a in range(A):
    for s in range(S):
        direct_cost = unit_costs[a,s]
        # print(f"Action {a}, State {s}, Direct cost: {direct_cost}")
        fail_risk = cs_pfs[s] * failure_cost
        cost = direct_cost + fail_risk
        R[a,s] = cost_util(cost, min_cost=0, max_cost=ma_cost)
# %%
#--------------------------------------------------------------------------------------------------------------------------------------------------------
# 2) Finite-horizon value iteration (vectorized, shape: P (A,S,S'), R (A,S))
def finite_horizon_value_iteration(P, R, gamma, H):
    """
    Simple value iteration for finite-horizon DP, without fancy numpy.
    - P: (A, S, S)  Transition matrices (A=actions, S=states)
    - R: (A, S)     Immediate reward for each (action, state)
    - gamma:        Discount factor
    - H:            Horizon (number of time steps)
    Returns:
      V_fun: (H+1, S)   Value function (at each time and state)
      Policy: (H, S)    Policy: best action at each time and state
    """

    # I need this because later on when I want to verify my code I need to get number of action and state inside the function
    na = P.shape[0]
    ncs = P.shape[1]

    # Initialize value and policy arrays
    V_fun = np.zeros((H+1, ncs), dtype=np.float64)  # H+1 because we need to store values from 0 to H
    Policy = np.zeros((H, ncs), dtype=np.int64)

    # Loop over each time step, backwards (last to first)
    for t in range(H-1, -1, -1): # starts at last time step(H-1, e.g. 199 if H=200) and stop before reaching -1 so include 0 and counts backward by 1
        for s in range(ncs):
            best_value = -np.inf
            best_action = 0
            for a in range(na):
                immediate_reward = R[a, s]
                future_expected_value = 0.0
                for s_next in range(ncs):
                    transition_prob = P[a, s, s_next]
                    future_expected_value += transition_prob * V_fun[t+1, s_next]
                # total value for taking action a in state s at time t
                total_value = immediate_reward + gamma * future_expected_value

                if total_value > best_value:
                    best_value = total_value
                    best_action = a

            # after loop over all actions, store the best one and its value
            V_fun[t, s] = best_value
            Policy[t, s] = best_action
    return V_fun, Policy

# %%
# Run DP value iteration
best_value, best_policy = finite_horizon_value_iteration(P, R, gamma, ELE_DP_HORIZON)

# %%
#-----------------------------------------------------------------------------------------------------------------------------------
# 3) Visualize DP results (values & policy over time)
fig, ax = plt.subplots(1,1, figsize=(7,4))
for s in range(ncs):
    ax.plot(best_value[:, s], label=f'CS{s+1}')
ax.set_title('DP Value function(end -> start)')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
plt.grid()
plt.show()

fig, ax = plt.subplots(1,1, figsize=(7,4))
for s in range(ncs):
    ax.plot(best_policy[:, s], label=f'CS{s+1}')
ax.set_title('DP Greedy Policy(end -> start)')
ax.set_xlabel('Time')
ax.set_ylabel('Action ID')
ax.legend()
plt.grid()
plt.show()
# %%
#--------------------------------------------------------------------------------------------------------------------------------------------------------
# 4)Verification of my code with simple example
print("Verification of my code with simple example:")
if __name__ == "__main__":
    # 2 states, 2 actions
    P_toy = np.zeros((2, 2, 2))
    P_toy[0] = [[0.8, 0.2], [0.0, 1.0]]
    P_toy[1] = [[0.5, 0.5], [0.2, 0.8]]
    R_toy = np.array([[5, 1], [10, -1]])
    gamma = 0.9
    H = 2

    V, Pi = finite_horizon_value_iteration(P_toy, R_toy, gamma, H)

    print("Compare the following results with hand-calculated values:")
    print("Test V:\n", V)
    print("Test Policy:\n", Pi)

# %%
