#  python -m dsm.scripts.make_dataset --env Pendulum-v1 
# --dataset_path datasets/pendulum/sac/dataset.pkl 
# --policy_params_path datasets/pendulum/sac/policy_params.msgpack --force

import os
import pathlib
import pickle
import logging
from typing import Callable, Annotated

import tyro
import functools
import numpy as np
import tqdm.rich as tqdm
import gymnasium as gym
import jax
import jax.numpy as jnp
import flax.serialization

from sbx import PPO, SAC
from sbx.common.policies import BaseJaxPolicy

from dsm import datasets, envs, stade
from dsm.types import Environment

# Define the type alias for the policy function
Policy = Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]

def make_sbx_model(env_id: Environment, env: gym.Env) -> PPO | SAC:
    match env_id:
        case "Pendulum-v1" | "dm_control/pendulum-swingup-v0":
            return SAC("MlpPolicy", env, verbose=1)
        case _:
            raise ValueError(f"Unknown environment: {env.spec!r}")

def load_policy(policy_params_path: pathlib.Path, model: PPO | SAC) -> Policy:
    # Load the model parameters
    with policy_params_path.open("rb") as f:
        params = flax.serialization.from_bytes(None, f.read())
    # Update the model's policy parameters
    model.policy.actor_state = model.policy.actor_state.replace(params=params)

    # Define the policy function
    def policy_fn(rng: jax.Array, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        rng, key = jax.random.split(rng)
        obs = jnp.expand_dims(observation, axis=0)
        action = BaseJaxPolicy.sample_action(model.policy.actor_state, obs, key)
        return rng, action

    return policy_fn
def main(
    dataset_path: pathlib.Path,
    *,
    env_id: Annotated[Environment, tyro.conf.arg(name="env")],
    seed: int | None = None,
    policy_params_path: pathlib.Path,
    # train_steps: int = 1_000,
    # num_eval_steps: int = 10,
    train_steps: int = 500_000,
    num_eval_steps: int = 500_000,
    sticky_action_prob: float = 0.0,
    force: bool = False,
):
    # Set up the environment for training (do not wrap)
    env = envs.make(env_id)

    # Initialize the model with the unwrapped environment
    model = make_sbx_model(env_id, env)

    # Training or loading the policy model
    if force or not policy_params_path.exists():
        # Train the model
        model.learn(total_timesteps=train_steps, progress_bar=True)

        # Save the model parameters
        params = model.policy.actor_state.params
        # Ensure the directory for policy parameters exists
        policy_params_path.parent.mkdir(parents=True, exist_ok=True)
        with policy_params_path.open("wb") as f:
            f.write(flax.serialization.to_bytes(params))
    else:
        # Load the model parameters
        with policy_params_path.open("rb") as f:
            params = flax.serialization.from_bytes(None, f.read())
        model.policy.actor_state = model.policy.actor_state.replace(params=params)

    # Wrap the environment for data collection
    env = stade.GymEnvWrapper(env, with_infos=False, seed=None)

    # Load the policy function
    policy_func = load_policy(policy_params_path, model)

    # Set up the RNG
    rng = jax.random.PRNGKey(seed if seed is not None else 0)

    action_t = None
    timestep_t = env.reset()
    transitions = [timestep_t]

    episode_index, episode_return = 0, 0.0
    for step in tqdm.tqdm(range(num_eval_steps)):
        rng, proposed_action = policy_func(rng, timestep_t.observation)

        # Apply sticky action logic
        if action_t is None or np.random.uniform() >= sticky_action_prob:
            action_t = proposed_action

        timestep_t = env.step(action_t)

        if not timestep_t.first():
            episode_return += timestep_t.reward
        if timestep_t.last():
            logging.info(f"Episode {episode_index} return: {episode_return}")
            episode_index += 1
            episode_return = 0.0

        timestep_t = timestep_t._replace(observation=jnp.squeeze(timestep_t.observation)) # changed
        transitions.append(timestep_t)

    print(f"Saving dataset with {len(transitions)} transitions.")
    # Convert transitions to numpy arrays and save
    # print(transitions)  
    # TimeStep(step_type=<StepType.MID: 1>,       reward=Array([-10.047763], dtype=float32),    discount=1.0,       
    #   observation=Array([-0.9417805 ,  0.33622837,  6.7871895 ], dtype=float32))]
    transitions = jax.tree_util.tree_map(lambda *arrs: np.vstack(arrs), *transitions)
    with dataset_path.open("wb") as fp:
        pickle.dump(transitions, fp)

    print("Program make_dataset.py done")


if __name__ == "__main__":
    tyro.cli(main)


# After training  - to get reward distribution- works with policy from make_dataset_pendulum_old.py - requires spec tensorflow versioning
# # Exploring actions and rewards of the 10 atoms for the samples above from  one source state

# # plotting utils
# from matplotlib.cm import ScalarMappable
# from matplotlib.colors import Normalize
# atom_idx_values = np.arange(samples.shape[0])
# norm = Normalize(vmin=min(atom_idx_values), vmax=max(atom_idx_values))
# sm = ScalarMappable(cmap='tab10', norm=norm)

# policy = datasets.make_policy(config.env)
# def apply_policy_to_batch(batch):
#     keys,samples = batch
#     return jax.lax.map(lambda x: policy(x[0],x[1]),(keys,samples))
# keys = jax.random.split(jax.random.PRNGKey(0), np.prod(samples.shape[:-1]))
# keys = jnp.array(keys).reshape(*samples.shape[:-1], -1)
# actions = jax.lax.map(apply_policy_to_batch, (keys,samples))[1]
# print('debug actions min max',np.min(actions), np.max(actions))

# # # Assuming actions[1] has shape (10, 64, 1, 1)
# # # Reshape it to (10, 64)
# # reshaped_actions = actions.reshape(actions.shape[0], -1)
# # # for i in range(10):
# # #     plt.scatter([i]*len(reshaped_actions[i]), reshaped_actions[i])
# # x_coords = np.repeat(np.arange(reshaped_actions.shape[0]), reshaped_actions.shape[1])
# # # Flatten reshaped_actions) for plotting
# # flat_actions = reshaped_actions.flatten()
# # plt.scatter(x_coords, flat_actions, c=x_coords, alpha=0.5, cmap='viridis')
# # plt.xlabel('Atom index')
# # plt.ylabel('Action value')
# # plt.show()

# rewards = jax.vmap(jax.vmap(reward_fn))(samples, actions).squeeze()

# reshaped_rewards = rewards.reshape(rewards.shape[0], -1)
# # for i in range(10):
# #     plt.scatter([i]*len(reshaped_actions[i]), reshaped_actions[i])
# x_coords = np.repeat(np.arange(reshaped_rewards.shape[0]), reshaped_rewards.shape[1])
# # Flatten reshaped_actions) for plotting
# flat_rewards = reshaped_rewards.flatten()
# plt.scatter(x_coords, flat_rewards, c=x_coords, alpha=0.5, cmap='viridis')
# plt.xlabel('Atom index')
# plt.ylabel('Reward value')
# plt.show()

# reward_mean = rewards.mean(axis=-1) / (1.0 - config.gamma)  
# reward_mean.shape
# import seaborn as sns
# sns.histplot(reward_mean, kde=True, color='blue')