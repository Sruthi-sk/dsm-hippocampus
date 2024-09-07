import functools
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dsm.types import Environment


# class WindyCartesianEnv(gym.Env):
#     """Windy grid world."""

#     def __init__(
#         self,
#         size: float = 10.0,
#         vel: float = 1.0,
#         wind_scale: float = 0.25,
#         max_steps: int = 200,
#         reward_fn: Callable[[np.ndarray, float], float] | None = None,
#     ):
#         super().__init__()
#         self.size = size
#         self.vel = vel
#         self.wind_scale = wind_scale
#         self.max_steps = max_steps
#         self.state = np.zeros(2)
#         self.clock = 0
#         if reward_fn is None:
#             reward_fn = self._default_reward_fn

#         self.reward_fn = reward_fn

#         self._action_to_vel = {
#             0: np.array([-1.0, 0.0]),  # Left
#             1: np.array([1.0, 0.0]),  # Right
#             2: np.array([0.0, -1.0]),  # Down
#             3: np.array([0.0, 1.0]),  # Up
#         }

#     def _default_reward_fn(self, x: np.ndarray, a: float) -> float:
#         return 0.0

#     @functools.cached_property
#     def observation_space(self) -> spaces.Box:
#         return spaces.Box(
#             shape=(2,),
#             dtype=np.float32,
#             low=-self.size * np.ones(2) / 2,
#             high=self.size * np.ones(2) / 2,
#         )

#     @functools.cached_property
#     def action_space(self) -> spaces.Discrete:
#         return spaces.Discrete(4)

#     def reset(
#         self,
#         *,
#         seed: int | None = None,
#         options: dict[Any, Any] | None = None,
#     ) -> tuple[Any, dict[Any, Any]]:
#         self.state = np.zeros(2)
#         self.clock = 0
#         return self.state, {}

#     def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[Any, Any]]:
#         r = self.reward_fn(self.state, action)
#         vel = self._action_to_vel[action] * self.vel
#         wind_dir = np.sign(self.state)
#         wind = np.random.uniform(size=(2,)) * self.wind_scale
#         self.state = (self.state + vel + wind * wind_dir).clip(-self.size / 2, self.size / 2)
#         self.clock += 1
#         done = self.clock >= self.max_steps

#         return self.state, r, False, done, {}


def generate_navigation_task_env(WALL = None):
    from ratinabox.contribs.TaskEnvironmentGym import (SpatialGoalEnvironment, SpatialGoal, Reward)
    from ratinabox.Agent import Agent
    # Make the environment and add a wall 
    DT = 0.1 # Time step
    T_TIMEOUT = 15 # Time out
    # GOAL_POS = np.array([0.5, 0.5]) # Goal position
    # GOAL_RADIUS = 0.1
    # REWARD = 1 # Reward
    REWARD_DURATION = 1 # Reward duration

    #LEARNING CONSTANTS
    TAU = 5 # Discount time horizon
    TAU_E = 5 # Eligibility trace time horizon
    ETA = 0.01 # Learning rate 
    
    env = SpatialGoalEnvironment(
        dt=DT,
        teleport_on_reset=True,
        episode_terminate_delay=REWARD_DURATION,)
    env.exploration_strength = 1 
    if WALL is not None: env.add_wall(WALL)

    # ratenv = SpatialGoalEnvironment(params={'dimensionality':'2D'},
    #                  render_every=1, # how often to draw on .render() 
    #                  teleport_on_reset=False, # teleport animal to new location each episode?
    #                  verbose=False)

    # #Make the reward which is given when a spatial goal is satisfied. Attached this goal to the environment
    # reward = Reward(REWARD,decay="none",expire_clock=REWARD_DURATION,dt=DT,)
    # goals = [SpatialGoal(env,pos=GOAL_POS,goal_radius=GOAL_RADIUS, reward=reward)]
    # # goals = [SpatialGoal(env), SpatialGoal(env),
    # #      SpatialGoal(env,pos=GOAL_POS,goal_radius=GOAL_RADIUS, reward=reward), SpatialGoal(env), SpatialGoal(env)]
    # env.goal_cache.reset_goals = goals 

    # #Recruit the agent and add it to environment
    ag = Agent(env,params={'dt':DT})
    env.add_agents(ag) #<-- this updates the agent creating an off by one error 

    return env #, ag


def make(env_id: Environment) -> gym.Env:
    match env_id:
        case "Pendulum-v1":
            return gym.make(env_id)
        case "WindyGridWorld-v0" | "WindyGridWorld-top-v0" | "WindyGridWorld-bottom-v0":
            return WindyCartesianEnv()
        case "Ratinabox-v0-pc":
            env = generate_navigation_task_env() # make the task environment and agent
            print('generated env', env_id)
            return env
        case "Ratinabox-v0-xy":
            env = generate_navigation_task_env() # make the task environment and agent
            print('generated env', env_id)
            return env
        case 'Ratinabox-v0-pc-random-walls':
            WALL = np.array([[0.4, 0], [0.4, 0.95]])
            env = generate_navigation_task_env(WALL)
            return env
        case _:
            # print('env_id:', env_id)
            if env_id.startswith("Ratinabox-v0-pc"):
                # Env = Environment() # not gym env  - or use TaskEnvironmentGym directly instead of SpatialGoal - todo: check
                env = generate_navigation_task_env()
                return env
            else:
        case _:
            raise NotImplementedError
