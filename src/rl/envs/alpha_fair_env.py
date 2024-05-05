"""
Create environment for RL agent.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.AllocationSolver import AllocationSolver

class AlphaFairEnvironment(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, solver: AllocationSolver, render_mode=None):

        # observation space is the tuple (time, supply, [demands so far], [allocations so far])
        max_demand = max([d.max() for d in solver.demand_distributions])

        low = [0, 0] + [0 for _ in range(solver.N)] + [0 for _ in range(solver.N)]
        high = [solver.N + 1, solver.max_supply_needed()] + [max_demand for _ in range(solver.N)] + [solver.max_supply_needed() for _ in range(solver.N)]
        self.observation_space = spaces.Box(low=np.array(low), high=np.array(high), shape=(len(low), ), seed=42, dtype=np.float32)

        # the action space is the budget allocation fraction
        self.action_space = spaces.Box(0, 1, shape=(1,), seed=42, dtype=np.float32)

        self.solver = solver

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        return np.array([self.t, self.supply] +
                        self.demands[:self.t + 1] + [0 for _ in range(self.solver.N - self.t - 1)] +
                        self.allocations[:self.t + 1] + [0 for _ in range(self.solver.N - self.t - 1)], dtype=np.float32)

    def _get_info(self):
        return {
            "envy": 0
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # store current state
        self.t = 0
        self.supply = self.solver.initial_supply
        # generate demands up front, reveal as we go
        # this assumes that the demands are independent
        self.demands = [d.sample(1)[0] for d in self.solver.demand_distributions]
        self.allocations = [0 for _ in range(self.solver.N)]

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # the action is a budget utilization fraction
        x = action[0] * self.supply
        self.supply -= x
        self.allocations[self.t] = x

        self.t += 1
        terminated = self.t == self.solver.N
        if not terminated:
            self.demands[self.t] = self.solver.demand_distributions[self.t].sample(1)[0]

        observation = self._get_obs()
        info = self._get_info()
        
        prev_social_welfare = self.solver.social_welfare(self.allocations[:self.t-1], self.demands[:self.t-1]) if self.t > 1 else 0
        current_social_welfare = self.solver.social_welfare(self.allocations[:self.t], self.demands[:self.t])
        reward = current_social_welfare - prev_social_welfare

        self.render()
        
        return observation, reward, terminated, False, info
    
    def render(self):
        if self.render_mode == "human":
            print(f"t = {self.t}, supply = {self.supply}, demands = {self.demands}, allocations = {self.allocations}")

        
        
        




