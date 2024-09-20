import jax
import jax.numpy as jnp

import os
import logging
import operator


from absl import app, flags 
# from absl import logging #as absl_logging
import orbax.checkpoint
from etils import epath
import fiddle as fdl
from fiddle.experimental import serialization

# import fiddle.extensions.jax
# from fiddle import absl_flags as fdl_flags
# import fiddle.absl_flags as fdl_flags

from dm_env import specs
from dsm import train
import typing
from typing import Any

from dsm.state import State
from dsm.state import FittedValueTrainState
from dsm.configs import Config
import numpy as np

import tqdm.rich as tqdm
import einops


def del_flags(FLAGS,key_del):
    flags_dict = FLAGS._flags()    
    keys_list = [keys for keys in flags_dict]
    # for keys in keys_list:
    #     FLAGS.__delattr__(keys)
    if key_del in keys_list:
            FLAGS.__delattr__(key_del)
            print('deleted ',key_del)

def _maybe_restore_state(checkpoint_manager: orbax.checkpoint.CheckpointManager, 
                         state: State,checkpoint_step_num) -> State:
    
    latest_step = checkpoint_manager.latest_step()
    # print('debug checkpoint_manager',checkpoint_manager)

    def _restore_state(step: int, directory: os.PathLike[str] | None = None) -> State:
        # print('Debug directory', directory)
        # logging.info(f"Restoring checkpoint from {directory or checkpoint_manager.directory} at step {step}.")
        restored = checkpoint_manager.restore(
            step,
            {"generator": state.generator, "discriminator": state.discriminator},
            directory=directory,
            # directory=os.path.abspath(directory or checkpoint_manager.directory),
        )
        [g_state, d_state] = operator.itemgetter("generator", "discriminator")(restored)
        return State(step=jnp.int32(step), generator=g_state, discriminator=d_state)
    
    if isinstance(checkpoint_step_num, int) and checkpoint_step_num in checkpoint_manager.all_steps():
        index = next((index for index, checkpoint in 
                      enumerate(checkpoint_manager._checkpoints) if checkpoint.step == checkpoint_step_num), None)
        selected_ckpt_step = checkpoint_manager._checkpoints[index].step
        return _restore_state(selected_ckpt_step)

    if latest_step:
        print('debug latest_step', latest_step)
        return _restore_state(latest_step)

    print('debug checkpoint_manager',checkpoint_manager)
    print('No checkpint found')
    # logging.info("No checkpoint found.")
    return state

config = None 
checkpoint_manager = None 

def load_checkpointmgr_and_config_from_path(model_path, env, checkpoint_step: int | None = None,):
    # load_state_and_config of dsm.train only loads generator of environments registered in gym
    # global config, checkpoint_manager 
    # if checkpoint_manager is None: 
    if 'workdir' not in flags.FLAGS:
        checkpoint_path = epath.DEFINE_path("workdir", model_path, "Working directory.")
        workdir = checkpoint_path.value  # workdir:: epath.Path =
    # jax.debug.print("DEBUG directory {bar}", bar=os.path.abspath(workdir))
    # config: Config = fdl.build(serialization.load_json((workdir / "config.json").read_text()))
    
    try:
        config = fdl.build(serialization.load_json((workdir / "config.json").read_text()))

        # checkpoint manager gets model_path and restores the checkpoint from the path
        checkpoint_manager = orbax.checkpoint.CheckpointManager(
        os.path.abspath(workdir),
        # checkpoint_path,
        checkpointers={
                "generator": orbax.checkpoint.PyTreeCheckpointer(),
                "discriminator": orbax.checkpoint.PyTreeCheckpointer(), # can comment out if unnecessary
        },
        options=orbax.checkpoint.CheckpointManagerOptions( enable_async_checkpointing=False, 
                                                        async_options=None,create=True, ),
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        del_flags(flags.FLAGS, 'workdir') 
    # del workdir, checkpoint_path
    del_flags(flags.FLAGS, 'workdir') 
    return config, checkpoint_manager


def load_model_state_and_config_from_checkpoint_dir(model_path, env, 
                                                    checkpoint_step: int | None = None,)-> tuple[FittedValueTrainState, Config]:
    
    # global config, checkpoint_manager 
    # if checkpoint_manager is None: 
    config, checkpoint_manager = load_checkpointmgr_and_config_from_path(model_path, env, checkpoint_step)
    # dir(checkpoint_manager)
    # checkpoint_manager.all_steps() # returns list of step numbers [5000, 25000, ...]
    # checkpoint_manager._checkpoints[1].step 

    rng = np.random.default_rng(config.seed)
    state_rng_key = jax.random.PRNGKey(rng.integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max))
    # Make dummy state
    state = train.make_state(state_rng_key, typing.cast(specs.DiscreteArray, env.observation_spec()), config)

    state = _maybe_restore_state(checkpoint_manager, state,checkpoint_step)
    return state, config


####################################################################################
# checkpoint manager gets model_path and restores the checkpoint from the path
# checkpoint_manager = orload_model_state_and_config_from_checkpoint_dirbax.checkpoint.CheckpointManager(
# os.path.abspath(workdir),
# checkpointers={
#         "generator": orbax.checkpoint.PyTreeCheckpointer(),
#         "discriminator": orbax.checkpoint.PyTreeCheckpointer(),
# },
# options=orbax.checkpoint.CheckpointManagerOptions( enable_async_checkpointing=False, async_options=None,create=True, ),
# )
# state = _maybe_restore_state(checkpoint_manager, state)
####################################################################################

# # compute_DSM_samples # #adapted from compute_return_distribution code
# saved_source_states = plotting.source_states(config.env) 


def compute_DSM_samples_single_source(
    state: FittedValueTrainState, #model generator
    rng: jax.Array,
    zs = None,
    *,
    config,
    source_state_current: list[float],
    num_samples = None,
): 
    # code adapted from plot_utils.return_distribution
    # for _stuff, source in zip(*saved_source_states): 
    # jax.debug.print("Selected Source {bar}", bar=source_state_current)
    if num_samples == None:
        num_samples = config.plot_num_samples  # Number of state samples 
    num_outer=config.num_outer # Number of model atoms   
    num_latent_dims=config.latent_dims # Dimension of input noise 
    # print('num_samples', num_samples,' num_outer (no of atoms):', num_outer, ' num_latent_dims ',num_latent_dims)
    
    # 'Simulating trajectories in an MDP'
    # Code from plot_utils.sample_from_sr # samples = plot_utils.sample_from_sr(...)
    # generates samples from the state representation - Generates samples from 
    # the model using the provided source state and configuration settings
    # source_state is used to create a context for sampling by repeating it across the number of samples and outer dimensions
    
    # rng not used if zs provided
    if zs is None:
        zs = jax.random.normal(rng, (num_samples, num_outer, num_latent_dims))
        # print('debugging generated zs min max: ',np.min(zs),np.max(zs), zs.shape)
    else: 
        assert zs.shape == (num_samples, num_outer, num_latent_dims)
        print('Using provided zs')
    context = einops.repeat(source_state_current, "s -> i o s", i=num_samples, o=num_outer)
    xs = jnp.concatenate((zs, context), axis=-1)
    ys = jax.vmap(state.apply_fn, in_axes=(None, 0))(state.params, xs)
    samples =  einops.rearrange(ys, "i o s -> o i s")
    # print('samples shape:',samples.shape) #num_outer, num_samples, 3
    
    #thetas = np.arctan2(samples[i, :, 1], samples[i, :, 0]) % (2 * np.pi)
    # velocities = samples[i, :, -1]
    return source_state_current, samples


def extract_params_ith_atom(model_generator, index: int, num_outer: int):  #self, 
    """
    Extract the parameters for the ith model out of `num_outer`.

    Args:
    - params: The parameter dictionary.
    - index: The index of the model (0-based).
    - num_outer: The number of outer dimensions.

    Returns:
    - A dictionary containing the parameters for the ith model.
    """
    atom_params = {}

    def extract_params_recursive(current_params, atom_params):
        if isinstance(current_params, dict):
            for key, value in current_params.items():
                if isinstance(value, dict):
                    atom_params[key] = {}
                    extract_params_recursive(value, atom_params[key])
                elif isinstance(value, (jax.Array, jnp.ndarray)) and value.shape[0] == num_outer:
                    atom_params[key] = value[index]
                else:
                    atom_params[key] = value

    extract_params_recursive(model_generator.params, atom_params)
    # print("Extracted params shapes for the i-th model:",jax.tree_map(jnp.shape, atom_params))
    return atom_params['params']['model']





#############################################################################################
# UNUSED FUNCTION
############################
# buildable from config
# fiddle.extensions.jax.enable()

# logging.getLogger("jax").setLevel(logging.INFO)
# jax.config.update("jax_numpy_rank_promotion", "raise")

# buildable = fdl_flags.create_buildable_from_flags(configs)
# logging.info(printing.as_str_flattened(buildable))
# config: configs.Config = fdl.build(buildable)
############################
# from absl.flags import FLAGS # deleted in windows
# define_string('fdl_config', None, 'The Fiddle configuration to use.')
# define_string('fdl_config_file', 'config_500.py', 'Path to the Fiddle configuration file.')

# import fiddle.absl_flags as fdl_flags
# FLAGS = flags.FLAGS
# # flags.DEFINE_string('fdl_config', 'base', 'The Fiddle configuration to use.')
# if not FLAGS.fdl_config and not FLAGS.fdl_config_file:
#         FLAGS.fdl_config = 'base'
# # _maybe_remove_absl_logger()

############################

# NUM_SECS = 1*60
# sel_dataset_positions = dataset_positions[:600]
# Ag.import_trajectory(times = np.linspace(0, NUM_SECS,len(sel_dataset_positions)), positions=sel_dataset_positions) #,interpolate=True
# for i in range(int(NUM_SECS / Ag.dt)):
#         Ag.update()
# Ag.plot_trajectory(color='changing',plot_head_direction=True)
# # Ag.plot_trajectory(framerate=1)

############################

# goals from log file
# import re
# goals = []
# with open(log_file, 'r') as f:
#     for line in f:
#         if re.search(r'my_logger - DEBUG - goals:', line):
#             # Extract the part of the line after 'goals:'
#             goal_str = line.split('goals: ')[1]
#             # Remove brackets and spaces
#             goal_str = goal_str.replace('[', '').replace(']', '').replace(' ', '')
#             # Split into individual arrays
#             goal_arrays = goal_str.split('),')
#             for goal_array in goal_arrays:
#                 # Remove 'array(' and ')' and split into coordinates
#                 coords = goal_array.replace('array(', '').replace(')', '').split(',')
#                 # Convert coordinates to floats and add to goals list
#                 goals.append([float(coord) for coord in coords])

# print(goals)
# goals = np.array(goals)
# plt.scatter(goals[:, 0], goals[:, 1], c='r', label='Goals')

# def print_params_shapes(params: Any, prefix: str = ""):
#         if isinstance(params, dict):
#                 for key, value in params.items():
#                         print_params_shapes(value, f"{prefix}.{key}" if prefix else key)
#         elif isinstance(params, (jax.Array, jnp.ndarray)):
#                 print(f"{prefix}: {params.shape}")
#         else:
#                 print(f"{prefix}: {type(params)}")

# print("Generator params shapes:")
# print_params_shapes(state.generator.params)
# print("\nDiscriminator params shapes:")
# print_params_shapes(state.discriminator.params)

# def print_main_keys(obj):
#         keys = vars(obj).keys()
#         print("\nMain keys of state:")
#         for key in keys:
#                 print(key)
# print_main_keys(state)



# capture_intermediates=True, mutable=["intermediates"])
