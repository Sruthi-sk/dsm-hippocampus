# python -m dsm.scripts.make_dataset --env Pendulum-v1 
# --dataset_path datasets/pendulum/sac/dataset.pkl --policy_path datasets/pendulum/sac/

import os
import tyro
import functools
import logging
import pathlib
import pickle
from typing import Annotated

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tqdm.rich as tqdm

from jax.experimental import jax2tf
from sbx import PPO, SAC
from sbx.common.policies import BaseJaxPolicy

from dsm import datasets, envs, stade
from dsm.types import Environment

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
# export TF_GPU_ALLOCATOR=cuda_malloc_async



def make_sbx_model(env_id: Environment, env: gym.Env) -> PPO | SAC:
    match env_id:
        case "Pendulum-v1" | "dm_control/pendulum-swingup-v0":
            return SAC("MlpPolicy", env, verbose=1)
        # case "MountainCarContinuous-v0":
        #     return SAC("MlpPolicy", env, verbose=1)
        # case "Ratinabox-v0":
        #     return SAC("MlpPolicy", env, verbose=1)
        # case "dm_control/walker-walk-v0":
        #     return SAC("MlpPolicy", env, verbose=1)
        # case "dm_control/hopper-stand-v0":
        #     return SAC("MlpPolicy", env, verbose=1)
        case _:
            raise ValueError(f"Unknown environment: {env.spec!r}")


def main(
    dataset_path: pathlib.Path,
    *,
    env_id: Annotated[Environment, tyro.conf.arg(name="env")],
    seed: int | None = None,
    # train_steps: int = 500_000,
    train_steps: int = 1_000,
    policy_path: pathlib.Path,
    # num_eval_steps: int = 500_000,
    num_eval_steps: int = 1_0,
    sticky_action_prob: float = 0.0,
    force: bool = False,
):
    env = envs.make(env_id)

    folder_dataset_path = os.path.dirname(dataset_path)
    # base_path = 'datasets/ratinaboxPCgoal/'  # dataset_path
    if not os.path.exists(folder_dataset_path):
        os.makedirs(folder_dataset_path)

    # Training the policy model
    if force or not policy_path.exists():
        model = make_sbx_model(env_id, env)
        model.learn(total_timesteps=train_steps, progress_bar=True)

        # tf.function converts policy_apply() to a TensorFlow graph that 
        # takes two arguments: 
        # an observation with shape of env.observation_space and an action 
        # represented as a tensor of shape (2,) and type np.uint32.
        # functools.partial allows to fix certain no of arguments
        
        # def policy_apply_jax(obs: np.ndarray, rng_key: jax.random.KeyArray) -> np.ndarray:
        #     obs = jnp.expand_dims(obs, axis=0)  # Expand observation dimension
        #     return BaseJaxPolicy.sample_action(model.policy.actor_state, obs, rng_key)
        
        # @jax.jit
        # def policy_fn(rng: jax.random.KeyArray, observation: jax.Array) -> tuple[jax.random.KeyArray, jax.Array]:
        #     rng, key = jax.random.split(rng)
        #     action = policy_apply_jax(jnp.squeeze(observation), key)
        #     return rng, action
    
        @functools.partial(
            tf.function,
            input_signature=[
                tf.TensorSpec(env.observation_space.shape, env.observation_space.dtype),  # pyright: ignore
                tf.TensorSpec((2,), np.uint32),  # pyright: ignore
            ],
        )
        @functools.partial(jax2tf.convert, with_gradient=False)
        def policy_apply(obs: np.ndarray, rng_key: np.ndarray) -> np.ndarray:
            obs = jnp.expand_dims(obs, axis=0) # changed
            return BaseJaxPolicy.sample_action(model.policy.actor_state, obs, rng_key)

        policy = tf.Module()
        policy.__call__ = policy_apply

        tf.saved_model.save(policy, policy_path.as_posix())

    env = stade.GymEnvWrapper(env, with_infos=False, seed=None)

    # Convert saved model to Jax function
    rng = np.random.default_rng(seed)
    rng_key = jax.random.PRNGKey(rng.integers(0, 2**32))
    policy_func = datasets.load_policy(policy_path)
    # policy_func = policy_fn 

    action_t = None
    timestep_t = env.reset()
    # print('debug timestep_t', timestep_t)
    transitions = [timestep_t]

    episode_index, episode_return = 0, 0.0
    for step in tqdm.tqdm(range(num_eval_steps)):
        # rng_key, proposed_action = policy_func(rng_key, timestep_t.observation)
        rng_key, proposed_action = policy_func(rng_key, jnp.squeeze(timestep_t.observation)) # changed
        
        if action_t is None or rng.uniform() >= sticky_action_prob:
            action_t = proposed_action
        timestep_t = env.step(action_t)  # pyright: ignore

        if not timestep_t.first():
            episode_return += timestep_t.reward  # pyright: ignore
        if timestep_t.last():
            logging.info(f"Episode {episode_index} return: {episode_return}")
            episode_index += 1
            episode_return = 0.0

        timestep_t = timestep_t._replace(observation=jnp.squeeze(timestep_t.observation)) # changed
        transitions.append(timestep_t)

    # print(transitions)  
    # TimeStep(step_type=<StepType.MID: 1>,       reward=Array([-10.047763], dtype=float32),    discount=1.0,       
    #   observation=Array([-0.9417805 ,  0.33622837,  6.7871895 ], dtype=float32))]
    print('saving dataset', len(transitions))
    transitions = jax.tree_util.tree_map(lambda *arrs: np.vstack(arrs), *transitions)
    with dataset_path.open("wb") as fp:
        pickle.dump(transitions, fp)

    print('Program make_dataset.py done')

if __name__ == "__main__":
    tyro.cli(main)
