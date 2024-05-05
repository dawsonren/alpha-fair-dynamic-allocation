import sys

from spinup import sac_pytorch as sac
import gym
import numpy as np
import torch
import torch.nn as nn

# add library to path (or else, src not visible)
sys.path.insert(0, "../")

from src.AllocationSolver import AllocationSolver
from src.dists import GammaDistribution
from src.rl2.actor_critic import ActorCritic

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
env_fn = gym.make("envs:envs/AlphaFairEnvironment-v0", solver=prob)

# Create the actor-critic module
actor_critic = ActorCritic(2 * prob.N + 2, 1)

logger_kwargs = dict(output_dir='rl2/logs', exp_name='expr-01')

sac(env_fn=env_fn,
    steps_per_epoch=5000,
    epochs=250,
    seed=1,
    lr=1e-3,
    batch_size=256,
    start_steps=1e5,
    update_every=5,
    update_after=1e5,
    num_test_episodes=200,
    logger_kwargs=logger_kwargs)