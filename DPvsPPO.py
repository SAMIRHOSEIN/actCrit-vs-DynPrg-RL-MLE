# %%
# Important: In DP, we need to just choose one condition state (CS) at each time step, not a distribution over CSs shows the sate of the system..
# 1) Imports
# DP(value iteration) vs PPO — Aligned Inputs
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib.patches import Patch
from scipy.stats import norm
from tensordict.nn import TensorDictModule
from torchrl.envs import GymWrapper

from bridge_gym.example_nbe107.cost_util import normalized_cost
from bridge_gym.example_nbe107.rl_env import SingleElement
from bridge_gym.example_nbe107.settings import ACTION_MODEL, CS_PFS, FAILURE_COST, NA, NCS, UNIT_COSTS
from softtree_ppo.training import PPOTrainer
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




# If inc_step=False, we  must not keep using best_policy[t, s]. we need a stationary best_policy[s]
def stationary_value_iteration(P, R, gamma, tol=1e-10, max_iter=100000):
    """
    Infinite-horizon discounted value iteration.
    - P: (A, S, S)
    - R: (A, S)
    Returns:
      V: (S,)
      Policy: (S,)
    """
    na = P.shape[0]
    ncs = P.shape[1]

    V = np.zeros(ncs, dtype=np.float64)

    for _ in range(max_iter):
        V_new = np.zeros_like(V)
        for s in range(ncs):
            q_vals = np.zeros(na, dtype=np.float64)
            for a in range(na):
                q_vals[a] = R[a, s] + gamma * np.dot(P[a, s], V)
            V_new[s] = np.max(q_vals)

        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new

    Policy = np.zeros(ncs, dtype=np.int64)
    for s in range(ncs):
        q_vals = np.zeros(na, dtype=np.float64)
        for a in range(na):
            q_vals[a] = R[a, s] + gamma * np.dot(P[a, s], V)
        Policy[s] = int(np.argmax(q_vals))

    return V, Policy
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
# 8) Define policy from DP table and evaluate in the PPO env
# DP action policy
class DPPolicyFiniteHorizon(nn.Module):
    """Torch module for finite-horizon DP policy: action = policy[t, s]."""
    def __init__(self, dp_table, ncs, horizon):
        super().__init__()
        self.register_buffer("dp_table", torch.as_tensor(dp_table, dtype=torch.long))
        self.ncs = ncs
        self.horizon = horizon

    def forward(self, observation):
        state_dis = observation[..., :self.ncs]
        tau = observation[..., self.ncs]

        s_idx = torch.argmax(state_dis, dim=-1)
        t_idx = torch.floor(tau * self.horizon).long()
        t_idx = torch.clamp(t_idx, min=0, max=self.dp_table.shape[0] - 1)

        action = self.dp_table[t_idx, s_idx]
        return action


class DPPolicyStationary(nn.Module):
    """Torch module for stationary DP policy: action = policy[s]."""
    def __init__(self, dp_table, ncs):
        super().__init__()
        self.register_buffer("dp_table", torch.as_tensor(dp_table, dtype=torch.long))
        self.ncs = ncs

    def forward(self, observation):
        state_dis = observation[..., :self.ncs]
        s_idx = torch.argmax(state_dis, dim=-1)
        action = self.dp_table[s_idx]
        return action



if __name__ == '__main__':

    # Inputs
    GAMMA = 1 / 1.03
    horizon = 5 #200

    # evaluation settings formerly in test_constants.py
    normalizer = 1.0
    ELE_DP_N_EPISODES = 10000

    ELE_DP_RESET_PROB = None
    ELE_DP_DIRICHLET_ALPHA = np.array([0.14964171, 0.11136174, 0.05003725, 0.03926025])
    ELE_DP_RANDOM_STATE = 42
    # ELE_DP_RESET_PROB = np.array([1.0, 0.0, 0.0, 0.0])
    # ELE_DP_DIRICHLET_ALPHA = None
    # ELE_DP_RANDOM_STATE = 'off'

    # True  -> finite-horizon DP, policy[t, s]
    # False -> stationary DP, policy[s]
    ELE_DP_INC_STEP = True



    horizon = horizon
    n_episodes = ELE_DP_N_EPISODES
    max_cost = normalizer
    reset_prob = ELE_DP_RESET_PROB
    dirichlet_alpha = ELE_DP_DIRICHLET_ALPHA
    random_state = ELE_DP_RANDOM_STATE
    include_step = ELE_DP_INC_STEP

    print("max_cost:", max_cost)
    print(f"include_step_count: {include_step}")
    print("reset_prob:", reset_prob)
    print("dirichlet_alpha:", dirichlet_alpha)
    print("random_state for env:", random_state)
    print("n_episodes:", n_episodes)
    print("horizon:", horizon)


    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    # 6)Run DP value iteration

    # Note: ACTION_MODEL[a][s', s] = P(s'|s,a)
    P = np.array([ACTION_MODEL[a] for a in range(NA)])  # (A, S, S)

    # Rewards (A, S): immediate reward at pure state s under action a
    R = np.zeros((NA, NCS), dtype=np.float64)

    for a in range(NA):
        for s in range(NCS):
            direct_cost = UNIT_COSTS[a, s]
            fail_risk = CS_PFS[s] * FAILURE_COST
            cost = direct_cost + fail_risk
            R[a, s] = normalized_cost(cost, normalizer=normalizer)

    if ELE_DP_INC_STEP:
        best_value, best_policy = finite_horizon_value_iteration(P, R, GAMMA, horizon)
        print("Finite-horizon DP selected")
        print("Shape of V = (Horizon+1, State) =", best_value.shape)
        print("Shape of Policy = (Horizon, State) =", best_policy.shape)
        print("Best policy table:")
        print(best_policy)
    else:
        best_value, best_policy = stationary_value_iteration(P, R, GAMMA)
        print("Stationary DP selected")
        print("Shape of V = (State,) =", best_value.shape)
        print("Shape of Policy = (State,) =", best_policy.shape)
        print("Best stationary policy:")
        print(best_policy)


    id2name = {
        0: "Do nothing",
        1: "Maintenance",
        2: "Repair",
        3: "Rehabilitation",
        4: "Replacement"
    }

    #-----------------------------------------------------------------------------------------------------------------------------------
    # 7) Visualize DP results (values & policy over time)
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    if ELE_DP_INC_STEP:
        for s in range(NCS):
            ax.plot(best_policy[:, s], label=f'CS{s+1}')
        ax.set_title('Finite-horizon DP Greedy Policy')
        ax.set_xlabel('Time')
        ax.legend()
    else:
        ax.bar(np.arange(NCS), best_policy)
        ax.set_title('Stationary DP Policy by State')
        ax.set_xlabel('State')
        ax.set_xticks(np.arange(NCS), [f'CS{i+1}' for i in range(NCS)])

    ax.set_ylabel('Action ID')
    ax.set_yticks(np.arange(0, 5, 1))
    plt.grid()
    plt.show()


    base_env = SingleElement(
        max_steps=horizon,
        discount=GAMMA,
        state_size=NCS,
        action_size=NA,
        include_step_count=include_step,
        reset_prob=reset_prob,
        dirichlet_alpha=dirichlet_alpha,
        action_model=ACTION_MODEL,
        unit_costs=UNIT_COSTS,
        pf_array=CS_PFS,
        failure_cost=FAILURE_COST,
        cost_kwargs={"normalizer": max_cost},
        seed=None if random_state == "off" else random_state,
    )
    env = GymWrapper(base_env, categorical_action_encoding=True)

    if include_step:
        dp_core = DPPolicyFiniteHorizon(best_policy, ncs=NCS, horizon=horizon)
    else:
        dp_core = DPPolicyStationary(best_policy, ncs=NCS)

    dp_actor = TensorDictModule(
        dp_core,
        in_keys=["observation"],
        out_keys=["action"],
    )

    logs = PPOTrainer.evaluate(
        actor=dp_actor,
        eval_env=env,
        num_episodes=n_episodes,
        max_steps=horizon + 1,
        deterministic=True,
        store_rollout=True,
    )

    print(f"Initial state: {logs['init_state'][0]}")
    print(f"Final state: {logs['observation'][0][-1]}")

    # DP evaluation summary stats for convergence analysis
    from eval_stats import mean_and_ci
    dp_stats = mean_and_ci(logs["eval_reward"], z=1.96)
    print(
        f"Parameters in Validation (episode return for {dp_stats['n']} episodes): "
        f"mean={dp_stats['mean']:.4f}, "
        f"95% CI=[{dp_stats['ci_low']:.4f}, {dp_stats['ci_high']:.4f}], "
        f"SD={dp_stats['sd']:.4f}, "
        f"N={dp_stats['n']}"
    )

    # Define the list of arrays
    all_actions = np.concatenate([np.asarray(a).reshape(-1) for a in logs["action"]])


    unique, counts = np.unique(all_actions, return_counts=True)
    action_distribution = {id2name.get(a, a): c for a, c in zip(unique, counts)}
    print(action_distribution)

    # --- Plot: initial beta vs episode LCC ---
    # logs["observation"] is a list, each element is an array with shape (T, obs_dim)
    init_obs = np.array(logs["init_state"])

    # if step-count is included, drop the last column so we keep only the condition-state vector
    # cs_pfs length is the number of condition states
    init_states = init_obs[:, :len(CS_PFS)]  # shape: (n_episodes, ncs)

    # state -> pf -> beta
    init_pf = init_states @ CS_PFS  # shape: (n_episodes,)
    init_beta = -norm.ppf(init_pf)  # shape: (n_episodes,)

    # LCC = total episode cost = negative of episode reward
    lcc_values = -np.array(logs["eval_reward"])

    # normalize LCC by episode length
    LCC_norm = lcc_values / horizon

    plot_df = pd.DataFrame({
        "initial_beta": init_beta,
        "LCC_norm": LCC_norm
    })

    csv_path = os.path.join(os.getcwd(), "initial_beta_vs_LCC_DP.csv")
    plot_df.to_csv(csv_path, index=False)

    print(f"Saved plot data to: {csv_path}")

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.scatterplot(x="initial_beta", y="LCC_norm", data=plot_df, ax=ax)

    ax.set_xlabel("β (Reliability Index)")
    ax.set_ylabel("LCC / max_steps")
    plt.title("Initial β vs LCC/max step (DP)")

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()



    # action distribution summary
    all_actions = np.concatenate([np.asarray(a).reshape(-1) for a in logs["action"]])
    id2name = {0:"Do nothing",1:"Maintenance",2:"Repair",3:"Rehabilitation",4:"Replacement"}
    unique, counts = np.unique(all_actions, return_counts=True)


    # ---- Action sequence for the single evaluation episode (t = 0..T-1 for steps actually taken) ----
    ep0_actions = logs["action"][0].astype(int).flatten()          # shape: (horizon,)
    ep0_obs     = logs["observation"][0]                           # shape: (horizon, obs_len)

    ep0_action_names = [id2name.get(int(a), str(int(a))) for a in ep0_actions]
    print("\nEvaluation action sequence (time-ordered):")
    print(", ".join(ep0_action_names))




    # --- Condition-state distribution per step (episode 0) ---
    ep0_obs = logs["observation"][0]                 # shape: (horizon, obs_dim)
    obs_dim = ep0_obs.shape[1]

    # If include_step_count=True, the last obs entry is normalized time; otherwise there is no time column.
    ncs_eff = int(obs_dim - (1 if include_step else 0))   # number of condition-state entries

    cs_traj = ep0_obs[:, :ncs_eff]                                # (horizon, ncs)
    t_traj  = ep0_obs[:, ncs_eff] if include_step else None

    print("\nCondition-state distribution per step (episode 0):")
    for t, (cs, a) in enumerate(zip(cs_traj, ep0_actions)):
        cs_str = ", ".join([f"cs{k}={p:.3f}" for k, p in enumerate(cs)])
        if t_traj is not None:
            # τ is the normalized time in [0,1]
            print(f"Step={t:02d}  t={t_traj[t]:.3f}  action={id2name[int(a)]:<14} [{cs_str}]")
        else:
            print(f"Step={t:02d}  act={id2name[int(a)]:<14} [{cs_str}]")





    # show the results for first 5 realizations for paper
    n_print = min(5, len(logs["observation"]))
    print(f"\nCondition-state distribution per step (first {n_print} episodes):")


    for ep in range(n_print):
        print(f"\n=== Episode {ep} ===")

        ep_obs = logs["observation"][ep]                 
        ep_actions = logs["action"][ep].astype(int).flatten()


        print(logs["reward"][ep].astype(float).flatten())
        ep_rewards = logs["reward"][ep].astype(float).flatten()   # shape: (horizon,)
        ep_return  = ep_rewards.sum()
        ep_avg     = ep_rewards.mean()

        print(f"Episode {ep}: return (sum) = {ep_return:.4f} | avg per step = {ep_avg:.6f}")



        obs_dim = ep_obs.shape[1]
        ncs_eff = int(obs_dim - (1 if include_step else 0))

        cs_traj = ep_obs[:, :ncs_eff]
        t_traj  = ep_obs[:, ncs_eff] if include_step else None

        for t, (cs, a) in enumerate(zip(cs_traj, ep_actions)):
            cs_str = ", ".join([f"cs{k}={p:.3f}" for k, p in enumerate(cs)])
            if t_traj is not None:
                print(
                    f"Step={t:02d}  "
                    f"t={t_traj[t]:.3f}  "
                    f"action={id2name[int(a)]:<14} "
                    f"[{cs_str}]"
                )
            else:
                print(
                    f"Step={t:02d}  "
                    f"action={id2name[int(a)]:<14} "
                    f"[{cs_str}]"
                )


    

    # Generated by AI
    # 1) action sequence for the single evaluation episode
    ep0_actions = logs["action"][0].astype(int).flatten()
    T = len(ep0_actions)

    # 2) color palette (crisp, colorblind-friendly-ish)
    action_colors = {
        0: "#9E9E9E",  # gray
        1: "#4E79A7",  # blue
        2: "#59A14F",  # green
        3: "#F28E2B",  # orange
        4: "#E15759",  # red
    }
    colors = [action_colors[int(a)] for a in ep0_actions]

    # 3) plot a single horizontal bar split into T segments
    fig, ax = plt.subplots(figsize=(max(10, T*0.35), 1.6))
    lefts = np.arange(T)
    ax.barh(
        y=0, width=np.ones(T), left=lefts, height=0.85,
        color=colors, edgecolor="white", linewidth=1.4
    )

    # 4) label each segment with the step number (1..T) centered, with auto-contrasting text
    for t, c in enumerate(colors):
        r, g, b = mcolors.to_rgb(c)
        luminance = 0.2126*r + 0.7152*g + 0.0722*b
        txt_color = "white" if luminance < 0.5 else "black"
        ax.text(t + 0.5, 0, str(t+1), ha="center", va="center", fontsize=9, color=txt_color, fontweight="bold")

    # 5) cosmetics: DP label on the left, no title, clean axes
    ax.set_ylim(-0.8, 0.8)
    ax.set_xlim(0, T)
    ax.set_yticks([0])
    ax.set_yticklabels(["DP"], fontsize=11)   # put "DP" on the left
    ax.set_xticks([])                         # numbers are inside each segment already
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    # optional legend (comment out if you don't want it)
    handles = [Patch(facecolor=action_colors[k], label=id2name[k]) for k in sorted(action_colors)]
    leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3, frameon=False)

    plt.tight_layout()
    plt.show()