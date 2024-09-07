"""
Dataset with states as place cells of the agent 
(from TaskEnv in RatInABox-v0) - 1 goal now in centre,

python -m dsm.scripts.make_dataset_rat_PC_goal --dataset_path datasets/ratinaboxPCgoal/sac/dataset.pkl --policy_path datasets/ratinaboxPCgoal/sac/policy --force 
- add to datasets.py
_DATASET_REGISTRY: 
    "Ratinabox-v0-pc-goal": pathlib.Path("datasets/ratinaboxPCgoal/sac/dataset.pkl"),
- change env in configs.py
- add to - dsm/plotting/_init_.py - _PLOT_BY_ENVIRONMENT 
python -m dsm --workdir logdir-rat_50pc_goal --fdl_config=base

"""

from ratinabox.contribs.TaskEnvironmentGym import (SpatialGoal, Reward)
from ratinabox.Agent import Agent
from ratinabox.Neurons import PlaceCells
import dsm.scripts.actor_critic_rat as rat_algos

import os
import tyro
import functools
import logging
import pathlib
import pickle
from typing import Annotated

TRAIN_STEPS = 4000
NUM_EVAL_STEPS = 5000
NUM_PLACE_CELLS = 50

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tqdm.rich as tqdm
import torch
from jax.experimental import jax2tf
from sbx import PPO, SAC
from sbx.common.policies import BaseJaxPolicy

from dsm import datasets, envs, stade
from dsm.types import Environment
import joblib

# def make_sbx_model(env_id: Environment, env: gym.Env) -> PPO | SAC:
#     match env_id:
#         case "Pendulum-v1" | "dm_control/pendulum-swingup-v0":
#             return SAC("MlpPolicy", env, verbose=1)

def main(
    dataset_path: pathlib.Path,
    *,
    # env_id: Annotated[Environment, tyro.conf.arg(name="env")],
    seed: int | None = None,
    train_steps: int = TRAIN_STEPS,
    # train_steps: int = 10,
    policy_path: pathlib.Path,
    num_eval_steps: int = NUM_EVAL_STEPS,
    # num_eval_steps: int = 10_0,
    sticky_action_prob: float = 0.0,
    force: bool = False,
):
    env_id = "Ratinabox-v0-pc-goal"

    folder_dataset_path = os.path.dirname(dataset_path)
    # base_path = 'datasets/ratinaboxPCgoal/'  # dataset_path
    if not os.path.exists(folder_dataset_path):
        os.makedirs(folder_dataset_path)
    logger = logging.getLogger('my_logger')
    # Set the level of the logger. This can be DEBUG, INFO, WARNING, ERROR, CRITICAL.
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(folder_dataset_path+'/my_log.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.debug(f'TRAIN_STEPS; {TRAIN_STEPS}, NUM_EVAL_STEPS: {NUM_EVAL_STEPS}')

    logger.debug(env_id)
    env = envs.make(env_id)
    DT = 0.1 # Time step

    #Make the reward which is given when a spatial goal is satisfied. Attached this goal to the environment
    REWARD = 1 # Reward
    REWARD_DURATION = 1 # Reward duration
    GOAL_POS = np.array([0.5, 0.5]) # Goal position
    GOAL_RADIUS = 0.1
    reward = Reward(REWARD,decay="none",expire_clock=REWARD_DURATION,dt=DT,)
    goals = [SpatialGoal(env,pos=GOAL_POS,goal_radius=GOAL_RADIUS, reward=reward)]
    # goals = [SpatialGoal(env), SpatialGoal(env),
    #      SpatialGoal(env,pos=GOAL_POS,goal_radius=GOAL_RADIUS, reward=reward), SpatialGoal(env), SpatialGoal(env)]
    env.goal_cache.reset_goals = goals 
    logger.debug(f'goals: {[goals[i].pos for i in range(len(goals))]}')
    
    # ag = Agent(env,params={'dt':DT}) # Agent already present when we made environment
    # env.add_agents(ag) #<-- this updates the agent creating an off by one error 
    # print(env.agent_names)
    ag = env.agent_lookup('agent_0')[0] 
    
    # Training the model
    # if force or not policy_path.exists():
    # if env_id!="Ratinabox-v0":
        #   model = make_sbx_model(env_id, env)
        #   model.learn(total_timesteps=train_steps, progress_bar=True)

    # # TODO - use last location of placecells if policy followed
    # pc_path = 'datasets/ratinaboxPCgoal/placecells_params_latest_run.pkl' 
    # PC_params = joblib.load(pc_path)
    # placecells = PlaceCells(Ag, params=PC_params) 
    placecells = PlaceCells(ag, params={'n':NUM_PLACE_CELLS,}) 
    logger.debug(f'placecells: {NUM_PLACE_CELLS}')

    #################################################################
    #Make the actor and the critic (first make their core NNs then pass these to the full Actor and Critic classes)
    actorNN  = rat_algos.VxVyGaussianMLP(n_in=placecells.n)
    criticNN = rat_algos.MultiLayerPerceptron(n_in=placecells.n)
    #
    actor  = rat_algos.Actor(ag, params = {'input_layers':[placecells], 'NeuralNetworkModule':actorNN}); actor.colormap="PiYG"
    critic = rat_algos.Critic(ag, params = {'input_layers':[placecells], 'NeuralNetworkModule':criticNN})

    #This visualises the value function and "policy" over the entire environment
    # fig, ax = critic.plot_rate_map(); fig.suptitle("Value function (before learning)")
    # fig, ax = actor.plot_rate_map(zero_center=True); fig.suptitle("Policy (before learning)"); ax[0].set_title("Vx"); ax[1].set_title("Vy")

    success_frac_list = []
    try: 
        for i in range(train_steps):  # train_steps or N_EPISODES
            # N_EPISODES = 10 # train_steps# Number of episodes   #TODO: 2000
            actor, critic = rat_algos.run_episode(env, 
                        ag,
                        actor, 
                        critic,
                        state_cells=[placecells],)
            success_frac = np.mean(np.array(env.episodes['meta_info'][-50:]) == "completed")
            success_frac_list.append(success_frac)
            episode_time = np.mean(env.episodes['duration'][-50:])
            print(f"{i}, <success fraction>: {success_frac:.2f}, <episode time> {episode_time:.1f}")
            if success_frac > 0.99 and i > 10: break
    except KeyboardInterrupt:
        print("Interrupted by user")

    # if success_frac < 0.9:
    #     print("Training did not converge. Exiting.")
    #     return

    logger.debug(f'success frac: {success_frac}')
    # print('debug env.observation_space.shape    ', env.observation_spaces[f'{env.agent_names[0]}']) # placecells.n

    # # setting policy to evaluate
    # if env_id == "Ratinabox-v0":

    # # converts a Python function to a TensorFlow graph to be saved
    # @functools.partial(
    #     tf.function,
    #     # input_signature=[
    #     #     tf.TensorSpec(env.observation_spaces[f'{env.agent_names[0]}'].shape, 
    #     #                   env.observation_spaces[f'{env.agent_names[0]}'].dtype),  # pyright: ignore
    #     #     tf.TensorSpec((2,), np.uint32),  # pyright: ignore
    #     # ],
    # )

    @functools.partial(jax2tf.convert, with_gradient=False)
    def policy_apply(obs: np.ndarray, rng_key: np.ndarray) -> np.ndarray:
        obs = jnp.expand_dims(obs, axis=0)
        # print('obs.shape ', obs.shape, 'obs', obs)
        # actor_fr = torch.tensor(actor_fr, requires_grad = True)
        # print('debug actor.firingrate_torch', actor_fr)
        action, _ = actor.NeuralNetworkModule.sample_action(actor.firingrate_torch)   # shouldnt i pass obs?
        return action
        # return actor.sample_action(actor.state, obs, rng_key)
        # return BaseJaxPolicy.sample_action(actor.state, obs, rng_key)

    policy = tf.Module()
    policy.__call__ = policy_apply
    # print('debug policy call ', policy, policy.__call__)

    tf.saved_model.save(policy, policy_path.as_posix())  # Or create path for dataset
    print('debug policy saved')
    ####################################################################

    env = stade.GymEnvWrapper(env, with_infos=False, seed=None)

    # Convert saved model to Jax function
    rng = np.random.default_rng(seed)
    rng_key = jax.random.PRNGKey(rng.integers(0, 2**32))
    # policy_func = datasets.load_policy(policy_path) # remove for ratinabox?

    action_t = None
    timestep_t = env.reset()
    # timestep_t = timestep_t._replace(observation=timestep_t.observation['agent_0'])

    cellfiringrates = placecells.get_history_arrays()                 # whole firing rate history?
    timestep_t = timestep_t._replace(observation=cellfiringrates["firingrate"][-1])
    # OR placecells.firingrate  OR placecells.get_state()

    transitions = [timestep_t] 
    # timestep_t.observation =timestep_t.observation['agent_0']

    episode_index, episode_return = 0, 0.0
    # print('debug timestep_t.observation', timestep_t, actor.firingrate_torch.detach().numpy())
    for step in tqdm.tqdm(range(num_eval_steps)):
        # rng_key, proposed_action = policy_func(rng_key, timestep_t.observation)
        # rng_key, proposed_action = policy_func(rng_key, jnp.squeeze(timestep_t.observation))

        # proposed_action = policy_apply(actor.firingrate_torch.detach().numpy(), timestep_t.observation, rng_key)
        proposed_action, _ = actor.NeuralNetworkModule.sample_action(actor.firingrate_torch) 

        if action_t is None or rng.uniform() >= sticky_action_prob:
            action_t = proposed_action
        # print('debug action_t ', action_t) # shape (2)
        timestep_t = env.step(action_t)  # pyright: ignore

        #######################
        state_cells=[placecells]
        # UPDATE THE STATE CELLS
        for cell in state_cells:
            cell.update()
        # UPDATE THE CRITIC AND ACTOR (INCLUDING LEARNING)
        critic.update(reward=timestep_t.reward, train=False)
        actor.update(td_error=critic.td_error,train=False)
        #######################
        if timestep_t.first():
            # timestep_t = timestep_t._replace(observation=timestep_t.observation['agent_0'])
            history_arrays = placecells.get_history_arrays()
            timestep_t = timestep_t._replace(observation=history_arrays["firingrate"][-1])
        if not timestep_t.first():
            episode_return += timestep_t.reward  # pyright: ignore
        if timestep_t.last():
            logging.info(f"Episode {episode_index} return: {episode_return}")
            episode_index += 1
            episode_return = 0.0
        cellfiringrates = placecells.get_history_arrays()["firingrate"][-1]
        timestep_t = timestep_t._replace(observation=cellfiringrates  )
        # print('debug timestep_t.observation', timestep_t)
        timestep_t = timestep_t._replace(observation=jnp.squeeze(timestep_t.observation))
        # timestep_t = timestep_t._replace(observation=timestep_t.observation['agent_0'])
        transitions.append(timestep_t)

    # print(transitions)  
    # TimeStep(step_type=<StepType.MID: 1>,       reward=Array([-10.047763], dtype=float32),    discount=1.0,       
    #   observation=Array([-0.9417805 ,  0.33622837,  6.7871895 ], dtype=float32))]

    transitions = jax.tree_util.tree_map(lambda *arrs: np.vstack(arrs), *transitions)
    with dataset_path.open("wb") as fp:
        pickle.dump(transitions, fp)
    
    # print('placecells.firingrate',placecells.firingrate)
    # print('placecells.get_state()',placecells.get_state()) 

    import ratinabox
    ratinabox.figure_directory = folder_dataset_path+'/figures'
    joblib.dump(placecells.params, folder_dataset_path+'/placecells_params.pkl')
    print('DEBUG cellfiringrates',placecells.get_history_arrays())
    placecells.plot_rate_map(autosave=True,method="history")
    ag.plot_trajectory(color="changing", autosave=True)

if __name__ == "__main__":
    tyro.cli(main)
