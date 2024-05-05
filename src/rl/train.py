import random
import sys

# add library to path (or else, src not visible)
sys.path.insert(0, "../")

import numpy as np
import gymnasium as gym
from tqdm import tqdm
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.rl.old_agents import DDPG, SAC
from src.AllocationSolver import AllocationSolver
from src.dists import GammaDistribution
from src.rl.actor_critic import ActorCritic
from src.rl.agents.sac import sac

# Create the problem instance
prob = AllocationSolver(
    [
        GammaDistribution(5, 3),
        GammaDistribution(3, 1),
        GammaDistribution(2, 1)
    ],
    10,
    alpha=np.inf
)

# Create and wrap the environment
env_fn = lambda: gym.wrappers.RecordEpisodeStatistics(gym.make("envs:envs/AlphaFairEnvironment-v0", solver=prob), 50)

# Create the actor-critic module
actor_critic = ActorCritic(2 * prob.N + 2, 1, hidden_size=16)

sac(env_fn=env_fn,
    ac_kwargs=dict(hidden_sizes=[16, 16], activation=torch.nn.ReLU, final_pi_activation=torch.nn.Sigmoid),
    steps_per_epoch=5000,
    epochs=250,
    seed=1,
    lr=1e-3,
    batch_size=256,
    start_steps=1e5,
    update_every=5,
    update_after=1e5,
    num_test_episodes=200)

"""
total_num_episodes = int(1e5)  # Total number of episodes
batch_size = 512
obs_space_dims = len(env.observation_space) # since a dict
action_space_dims = env.action_space.shape[0]
rewards_over_seeds = []

def map_dict_to_tensor(obs):
    return torch.tensor(np.array([obs["time"], obs["supply"]] + list(obs["demands"]) + list(obs["allocations"])), dtype=torch.float32)

def fibonacci_seeds(n):
    # give the first n fibonacci numbers
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

for seed in tqdm(fibonacci_seeds(1)):  # Fibonacci seeds
    # set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Reinitialize agent every seed
    # agent = DDPG(prob.N,
    #              actor_learning_rate=1e-3,
    #              critic_learning_rate=1e-2,
    #              gamma=0.99,
    #              tau=1e-2,
    #              memory_max_size=5000)
    agent = SAC(prob.N,
                lr=0.0001,
                memory_max_size=80000,
                tau=0.005,
                update_interval=5,
                target_update_interval=10000
                )
    reward_over_episodes = []

    for episode in tqdm(range(1, total_num_episodes + 1)):
        # gymnasium v26 requires users to set seed while resetting the environment
        obs, info = wrapped_env.reset(seed=seed)

        states = [obs]
        rewards = []

        done = False
        while not done:
            action = agent.select_action(map_dict_to_tensor(obs))

            # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
            # These represent the next observation, the reward from the step,
            # if the episode is terminated, if the episode is truncated and
            # additional info from the step
            next_obs, reward, terminated, truncated, info = wrapped_env.step(action)
            agent.rewards.append(reward)
            agent.memory.push(map_dict_to_tensor(obs), action, reward, map_dict_to_tensor(next_obs), terminated)

            obs = next_obs

            # End the episode when either truncated or terminated is true
            #  - truncated: The episode duration reaches max number of timesteps
            #  - terminated: Any of the state space values is no longer finite.
            done = terminated or truncated

            states.append(obs)
            rewards.append(reward)

        if episode % 200 == 0:
            print(states, rewards)

        reward_over_episodes.append(wrapped_env.return_queue[-1])

        if len(agent.memory) > batch_size:
            agent.update(batch_size)

        if episode % 1000 == 0:
            avg_reward = int(np.mean(wrapped_env.return_queue))
            print("Episode:", episode, "Average Reward:", avg_reward)

    rewards_over_seeds.append(reward_over_episodes)

rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
df1 = pd.DataFrame(rewards_to_plot).melt()
df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
sns.set_theme(style="darkgrid", context="talk", palette="rainbow")
sns.lineplot(x="episodes", y="reward", data=df1).set(
    title="DDPG for Alpha-Fair Problem"
)
plt.show()
"""