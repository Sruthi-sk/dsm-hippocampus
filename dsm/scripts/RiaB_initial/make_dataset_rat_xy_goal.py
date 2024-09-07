"""
Dataset with states as just x nd y positions of the agent ( from TaskEnv in RatInABox-v0)
"""

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
import tyro
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
#         case "Ratinabox-v0":
#             return SAC("MlpPolicy", env, verbose=1)
#         case "dm_control/walker-walk-v0":
#             return SAC("MlpPolicy", env, verbose=1)
#         case "dm_control/hopper-stand-v0":
#             return SAC("MlpPolicy", env, verbose=1)
#         case _:
#             raise ValueError(f"Unknown environment: {env.spec!r}")

def generate_navigation_task_env():
    from ratinabox.contribs.TaskEnvironmentGym import (SpatialGoalEnvironment, SpatialGoal, Reward)
    from ratinabox.Agent import Agent
    # Make the environment and add a wall 
    DT = 0.1 # Time step
    T_TIMEOUT = 15 # Time out
    GOAL_POS = np.array([0.5, 0.5]) # Goal position
    WALL = None
    GOAL_RADIUS = 0.1
    REWARD = 1 # Reward
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
    #Make the reward which is given when a spatial goal is satisfied. Attached this goal to the environment
    reward = Reward(REWARD,decay="none",expire_clock=REWARD_DURATION,dt=DT,)
    goals = [SpatialGoal(env,pos=GOAL_POS,goal_radius=GOAL_RADIUS, reward=reward)]
    # goals = [SpatialGoal(env), SpatialGoal(env),
    #      SpatialGoal(env,pos=GOAL_POS,goal_radius=GOAL_RADIUS, reward=reward), SpatialGoal(env), SpatialGoal(env)]
    
    env.goal_cache.reset_goals = goals 
    #Recruit the agent and add it to environment
    ag = Agent(env,params={'dt':DT})
    env.add_agents(ag) #<-- this updates the agent creating an off by one error 

    return env, ag

def main(
    dataset_path: pathlib.Path,
    *,
    env_id: Annotated[Environment, tyro.conf.arg(name="env")],
    seed: int | None = None,
    train_steps: int = 300,
    # train_steps: int = 10,
    policy_path: pathlib.Path,
    num_eval_steps: int = 500,
    # num_eval_steps: int = 10_0,
    sticky_action_prob: float = 0.0,
    force: bool = False,
):
    assert env_id == "Ratinabox-v0-xy"
    if env_id == "Ratinabox-v0-xy":
            # from ratinabox.contribs.TaskEnvironmentGym import (SpatialGoalEnvironment, TaskEnvironmentGym, 
            # SpatialGoal, Reward, Goal, get_goal_vector)
            
            # ratenv = SpatialGoalEnvironment(params={'dimensionality':'2D'},
            #                  render_every=1, # how often to draw on .render() 
            #                  teleport_on_reset=False, # teleport animal to new location each episode?
            #                  verbose=False)
            print(env_id)
            #TASK CONSTANTS
           
            N_EPISODES = 10 # train_steps# Number of episodes   #TODO: 2000
            L2 = 0.000 # L2 regularization
            env, ag = generate_navigation_task_env() # make the task environment and agent
            print('generated env')
            # return gym.make(env_id)
    else:
        env = envs.make(env_id)

    # Training the model
    # if force or not policy_path.exists():
    if env_id == "Ratinabox-v0-xy":
        import dsm.scripts.actor_critic_rat as rat_algos
        from ratinabox.Neurons import PlaceCells
        placecells = PlaceCells(ag, params={'n':50,}) 

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

        # else:
        #     model = make_sbx_model(env_id, env)
        #     model.learn(total_timesteps=train_steps, progress_bar=True)

    
        # if success_frac < 0.9:
        #     print("Training did not converge. Exiting.")
        #     return

        # # setting policy to evaluate
        if env_id == "Ratinabox-v0-xy":
        #     # @functools.partial(
        #     #     tf.function,
        #     #     input_signature=[
        #     #         tf.TensorSpec(env.observation_spaces[f'{env.agent_names[0]}'].shape, 
        #     #                       env.observation_spaces[f'{env.agent_names[0]}'].dtype),  # pyright: ignore
        #     #         tf.TensorSpec((2,), np.uint32),  # pyright: ignore
        #     #     ],
        #     # )

            print('debug env.observation_space.shape    ', env.observation_spaces[f'{env.agent_names[0]}'])
            @functools.partial(jax2tf.convert, with_gradient=False)
            def policy_apply(obs: np.ndarray, rng_key: np.ndarray) -> np.ndarray:
                obs = jnp.expand_dims(obs, axis=0)
                # print('obs.shape ', obs.shape, 'obs', obs)
                # actor_fr = torch.tensor(actor_fr, requires_grad = True)
                # print('debug actor.firingrate_torch', actor_fr)
                action, _ = actor.NeuralNetworkModule.sample_action(actor.firingrate_torch)   # shouldnt i pass obs?
                return action
        #         # return actor.sample_action(actor.state, obs, rng_key)
        #         # return BaseJaxPolicy.sample_action(actor.state, obs, rng_key)

        # else:
        #     @functools.partial(
        #         tf.function,
        #         input_signature=[
        #             tf.TensorSpec(env.observation_space.shape, env.observation_space.dtype),  # pyright: ignore
        #             tf.TensorSpec((2,), np.uint32),  # pyright: ignore
        #         ],
        #     )
        #     @functools.partial(jax2tf.convert, with_gradient=False)
        #     def policy_apply(obs: np.ndarray, rng_key: np.ndarray) -> np.ndarray:
        #         obs = jnp.expand_dims(obs, axis=0)
        #         return BaseJaxPolicy.sample_action(model.policy.actor_state, obs, rng_key)

        policy = tf.Module()
        policy.__call__ = policy_apply
        # print('debug policy call ', policy, policy.__call__)

        tf.saved_model.save(policy, policy_path.as_posix())  # Or create path for dataset
        print('debug policy saved')

    env = stade.GymEnvWrapper(env, with_infos=False, seed=None)

    # Convert saved model to Jax function
    rng = np.random.default_rng(seed)
    rng_key = jax.random.PRNGKey(rng.integers(0, 2**32))
    # if env_id != "Ratinabox-v0":
        # policy_func = datasets.load_policy(policy_path)

    action_t = None
    timestep_t = env.reset()
    timestep_t = timestep_t._replace(observation=timestep_t.observation['agent_0'])
    # cellfiringrates = placecells.get_history_arrays()["firingrate"][-1]                 # whole firing rate history?
    # timestep_t = timestep_t._replace(observation=cellfiringrates  )
    # OR placecells.firingrate  OR placecells.get_state()
    
    transitions = [timestep_t] 
    # timestep_t.observation =timestep_t.observation['agent_0']

    episode_index, episode_return = 0, 0.0
    # print('debug timestep_t.observation', timestep_t, actor.firingrate_torch.detach().numpy())
    for step in tqdm.tqdm(range(num_eval_steps)):
        # rng_key, proposed_action = policy_func(rng_key, timestep_t.observation)
        # rng_key, proposed_action = policy_func(rng_key, jnp.squeeze(timestep_t.observation))

        # proposed_action = policy_apply(jnp.squeeze(timestep_t.observation), rng_key)
        # proposed_action = policy_apply(actor.firingrate_torch.detach().numpy(), timestep_t.observation, rng_key)
        proposed_action, _ = actor.NeuralNetworkModule.sample_action(actor.firingrate_torch) 

        if action_t is None or rng.uniform() >= sticky_action_prob:
            action_t = proposed_action
        print('debug action_t ', action_t)
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
            timestep_t = timestep_t._replace(observation=timestep_t.observation['agent_0'])
            # history_arrays = placecells.get_history_arrays()
            # timestep_t = timestep_t._replace(observation=history_arrays["firingrate"][-1])

        if not timestep_t.first():
            episode_return += timestep_t.reward  # pyright: ignore
        if timestep_t.last():
            logging.info(f"Episode {episode_index} return: {episode_return}")
            episode_index += 1
            episode_return = 0.0

        # cellfiringrates = placecells.get_history_arrays()["firingrate"][-1]
        # timestep_t = timestep_t._replace(observation=cellfiringrates  )
        
        print('debug timestep_t.observation', timestep_t)
        timestep_t = timestep_t._replace(observation=jnp.squeeze(timestep_t.observation))
        # timestep_t = timestep_t._replace(observation=timestep_t.observation['agent_0'])
        transitions.append(timestep_t)

    # print(transitions)  
    # TimeStep(step_type=<StepType.MID: 1>,       reward=Array([-10.047763], dtype=float32),    discount=1.0,       
    #   observation=Array([-0.9417805 ,  0.33622837,  6.7871895 ], dtype=float32))]

    transitions = jax.tree_util.tree_map(lambda *arrs: np.vstack(arrs), *transitions)
    with dataset_path.open("wb") as fp:
        pickle.dump(transitions, fp)
    
    print('placecells.firingrate',placecells.firingrate)
    print('placecells.get_state()',placecells.get_state()) 

    # activations_alllayers = joblib.load('activations_alllayers_latent0.pkl')
    print('Program make_dataset.py done - success frac: ', success_frac)
    import ratinabox
    ratinabox.figure_directory = '/home/sruthi/Documents/thesis/distributional-sr/figures'
    placecells.plot_rate_map(autosave=True)
    joblib.dump(placecells.params, 'placecells_params_latest_run.pkl')

if __name__ == "__main__":
    tyro.cli(main)
