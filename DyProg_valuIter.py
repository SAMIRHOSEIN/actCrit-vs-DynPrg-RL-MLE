# %%
# 1) Imports
# DP(value iteration) vs PPO — Aligned Inputs
import numpy as np
import matplotlib.pyplot as plt

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
#--------------------------------------------------------------------------------------------------------------------------------------------------------
# 3)Verification of my code with simple example
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


    # i want to show structured tableof this:
    print("The output of my hand calculation:")
    print("| t | state | V[t,state] | best action (policy) |\n"
          "| - | ----- | ----------- | ------------------- |\n"
            "| 2 | 0     | 0           | -                   |\n"
            "| 2 | 1     | 0           | -                   |\n"
            "| 1 | 0     | 10          | 1                   |\n"
            "| 1 | 1     | 1           | 0                   |\n"
            "| 0 | 0     | 14.95       | 1                   |\n"
            "| 0 | 1     | 1.9         | 0                   |\n")


    print("Compare the following results with hand-calculated values:")
    print("Test V:\n", V)
    print("Test Policy:\n", Pi)
    print("Shape of V = (Horizon+1, State) =", V.shape)
    print("Shape of Policy = (Horizon, State) =", Pi.shape)

# %%
#--------------------------------------------------------------------------------------------------------------------------------------------------------
# 4) Align DP with PPO environment              
# This notebook aligns Dynamic Programming (finite-horizon value iteration) with our PPO environment so both share:
# - the same state/action spaces (ncs, na),
# - the same transition matrices (action_model),
# - the same reward/cost definition (unit_costs, cs_pfs, failure_cost),
# - the same discount factor gamma and horizon.

# Source of truth: element/problem_setup.py and element/rl_env.py.
import os
import element.problem_setup
import test_constants
from element.utility_func import cost_util
from element.rl_env import SingleElement
# Initialize environment to access its parameters
# load constants
A = element.problem_setup.na  # number of actions
S = element.problem_setup.ncs  # number of  state
gamma = element.problem_setup.gamma
action_model = element.problem_setup.action_model
unit_costs = element.problem_setup.unit_costs
failure_cost = element.problem_setup.failure_cost
cs_pfs = element.problem_setup.cs_pfs


max_cost_ = test_constants.ELE_DP_MAX_COST
horizon = test_constants.ELE_DP_HORIZON
reset_prob = test_constants.ELE_DP_RESET_PROB

#--------------------------------------------------------------------------------------------------------------------------------------------------------
# 5) Build DP model directly from the PPO env parameters
# We construct:
# - P with shape (A, S, S) using action_model[a] (rows: next state, cols: current state), consistent with `env` where next = A[a].T @ state` ⇒ column-wise transitions.
# - R with shape (A, S) using the same cost definition used in the env step:
#   # cost = unit_costs + failure_cost * cs_pfs
#   and reward via `cost_util(cost, min_cost=0, max_cost=unit_costs.max())`.
#Build P and R from env constants


# Note: action_model[a][s', s] = P(s'|s,a)
P = np.array([action_model[a] for a in range(A)])  # (A, S, S)

# Rewards (A, S): immediate reward at pure state s under action a
ma_cost = max_cost_
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
# 6)Run DP value iteration
best_value, best_policy = finite_horizon_value_iteration(P, R, gamma, horizon)
print("Shape of V = (Horizon+1, State) =", best_value.shape)
print("Shape of Policy = (Horizon, State) =", best_policy.shape)
# %%
#-----------------------------------------------------------------------------------------------------------------------------------
# 7) Visualize DP results (values & policy over time)
fig, ax = plt.subplots(1,1, figsize=(7,4))
for s in range(S):
    ax.plot(best_value[:, s], label=f'CS{s+1}')
ax.set_title('DP Value function(end -> start)')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
plt.grid()
plt.show()

fig, ax = plt.subplots(1,1, figsize=(7,4))
for s in range(S):
    ax.plot(best_policy[:, s], label=f'CS{s+1}')
ax.set_title('DP Greedy Policy(end -> start)')
ax.set_xlabel('Time')
ax.set_ylabel('Action ID')
ax.legend()
plt.grid()
plt.show()

# %% 
#--------------------------------------------------------------------------------------------------------------------------------------------------------
# 8) Define policy from DP table and evaluate in the PPO env
import numpy as np
import torch

from torchrl_bridge import create_element_env

from tqdm import tqdm
from collections import defaultdict
from torchrl.envs.utils import set_exploration_type

# DP action policy
def action_policy_dp(obs, t_idx, DPtable, ncs):
    """
    Pick DP action for the current observation and time index.
    - obs: observation, shape (ncs,) not (ncs+1,)
    - t_idx: current time index
    - DPtable: (H, ncs) best_action[t, s]
    """
    # choose the CS based on the most probable CS
    state_dis = obs[:ncs]
    s = int(np.argmax(state_dis))  

    # Greedy DP action
    return int(DPtable[t_idx, s])


if __name__ == '__main__':
    import importlib
    import test_constants
    importlib.reload(test_constants)

    # load constants
    reset_prob = test_constants.ELE_DP_RESET_PROB
    horizon = test_constants.ELE_DP_HORIZON
    n_episodes = test_constants.ELE_DP_N_EPISODES
    max_cost = test_constants.ELE_DP_MAX_COST
    reset_prob = test_constants.ELE_DP_RESET_PROB
    dirichlet_alpha = test_constants.ELE_DP_DIRICHLET_ALPHA
    random_state = test_constants.ELE_DP_RANDOM_STATE
    explore_type = test_constants.ELE_DP_EXPLORE_TYPE
    include_step = test_constants.ELE_DP_INC_STEP

    # recreate env
    env = create_element_env(
        horizon,
        max_cost=max_cost,
        include_step_count=include_step,
        reset_prob=reset_prob,
        dirichlet_alpha=dirichlet_alpha,
        random_state=random_state
    )

    # gather experience
    logs = defaultdict(list)
    eval_str = ""


    with tqdm(total=n_episodes*horizon) as pbar:
        with set_exploration_type(explore_type), torch.no_grad():
            for _ in range(n_episodes):


                td = env.reset()
                # figure observation length at runtime (ncs + 1 if include_step_count)
                obs_len = int(td["observation"].numel())
                print(f"Observation length: {obs_len}")
                print(f"include_step_count: {include_step}")
                observation = np.zeros((horizon, obs_len), dtype=np.float32)
                action      = np.zeros((horizon, 1),   dtype=np.int64)
                reward      = np.zeros((horizon, 1),   dtype=np.float32)


                # show initial obs/time
                init_obs = td["observation"]
                if include_step:
                    init_time_idx = int(init_obs[-1].item() * horizon)
                    print("Initial time index:", init_time_idx)

                print("Initial observation:", init_obs)


                # rollout
                for i in range(init_time_idx, horizon):
                    curr_obs = td["observation"]

                    a = action_policy_dp(curr_obs, i, best_policy, S)
                    td["action"] = torch.tensor(a, dtype=torch.int64)

                    res = env.step(td)

                    observation[i] = res["observation"].cpu().numpy()
                    action[i]   = res["action"].cpu().numpy()
                    reward[i]   = res["next", "reward"].cpu().numpy()
                    td["observation"] = res["next", "observation"]
                    
                    
                # log rollout data
                logs["observation"].append(observation)
                logs["action"].append(action)
                logs["reward"].append(reward)
                logs["ep reward"].append(reward.sum().item())

                eval_str = (
                    f"ep reward: {logs['ep reward'][-1]: 4.4f} "
                    f"(init: {logs['ep reward'][0]: 4.4f}), "
                )
                pbar.set_description(eval_str)
                pbar.update(horizon)
    
    print(f"Initial state: {logs['observation'][0][0]}")
    print(f"Final state: {logs['observation'][0][-1]}")
    print(f"Average ep reward: {np.mean(logs['ep reward']): 4.4f}")
    
    # Define the list of arrays
    all_actions = logs["action"]
    all_actions = np.concatenate(all_actions)


    id2name = {
        0: "Do nothing",
        1: "Maintenance",
        2: "Repair",
        3: "Rehabilitation",
        4: "Replacement"
    }
    unique, counts = np.unique(all_actions, return_counts=True)
    action_distribution = {id2name.get(a, a): c for a, c in zip(unique, counts)}
    print(action_distribution)



# %%
