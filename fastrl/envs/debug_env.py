# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/03_Environment/06_envs.debug_env.ipynb.

# %% auto 0
__all__ = ['SimpleContinuousEnv']

# %% ../../nbs/03_Environment/06_envs.debug_env.ipynb 2
# Python native modules
import os
# Third party libs
import gymnasium as gym
from gymnasium import spaces
import numpy as np
# Local modules

# %% ../../nbs/03_Environment/06_envs.debug_env.ipynb 4
class SimpleContinuousEnv(gym.Env):
    metadata = {'render.modes': ['console']}
    
    def __init__(self, goal_position=None, proximity_threshold=0.5):
        super(SimpleContinuousEnv, self).__init__()
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32)
        
        self.goal_position = goal_position if goal_position is not None else np.random.uniform(-10, 10)
        self.proximity_threshold = proximity_threshold
        self.state = None

    def step(self, action):
        self.state += action
        
        # Calculate the distance to the goal
        distance_to_goal = np.abs(self.state - self.goal_position)
        
        # Calculate reward: higher for being closer to the goal
        reward = -distance_to_goal
        
        # Check if the agent is within the proximity threshold of the goal
        done = distance_to_goal <= self.proximity_threshold
        
        info = {}
        
        return self.state, reward, done, info

    def reset(self):
        self.state = np.array([0.0], dtype=np.float32)
        if self.goal_position is None:
            self.goal_position = np.random.uniform(-10, 10)
        return self.state


    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError("Only console mode is supported.")
        print(f"Position: {self.state} Goal: {self.goal_position}")


