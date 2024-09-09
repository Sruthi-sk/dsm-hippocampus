# python load_model_compute_distr.py --fdl_config=base 

import contextlib
import inspect
import logging
import operator
import os
import sys
import typing
from typing import Any

import fancyflags as ff
import fiddle as fdl
import fiddle.extensions.jax
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint
import tqdm.rich as tqdm
from absl import app, flags
from absl import logging as absl_logging
from clu import metric_writers
from dm_env import specs
from etils import epath
from fiddle import absl_flags as fdl_flags
from fiddle import printing
from fiddle.codegen import codegen
from fiddle.experimental import serialization
from flax import traverse_util

from dsm import console, datasets, envs, stade, train
from dsm.state import State

# tf2jax.update_config('xlacallmodule_strict_checks', False)

#TODO Alter the contents of the 'configs' file to that of 'configs_pendulum' or any other appropriate file. Avoid directly importing this file because numerous internal files utilize 'config' for their component initialization.
from dsm import configs   
from dsm import metrics

_WORKDIR = epath.DEFINE_path("workdir", "logdir_pendulum_200k", "Working directory.")  ## change if different env

# _WORKDIR = epath.DEFINE_path("workdir", "logdir-500k_pendulum", "Working directory.")  ## change if different env
_CHECKPOINT_FROM = flags.DEFINE_string(
    "checkpoint_from",
    None,
    "Checkpoint to load from, we'll only restore from this checkpoint if "
    "the checkpoint step is greater than the current step."
    "If not specified, will load from the latest checkpoint in the working directory.",
)
_PROFILE = flags.DEFINE_bool("profile", False, "Enable profiling.")


def _maybe_restore_state(checkpoint_manager: orbax.checkpoint.CheckpointManager, state: State) -> State:
    latest_step = checkpoint_manager.latest_step()

    def _restore_state(step: int, directory: os.PathLike[str] | None = None) -> State:
        print('Debug directory', directory)
        logging.info(f"Restoring checkpoint from {directory or checkpoint_manager.directory} at step {step}.")
        restored = checkpoint_manager.restore(
            step,
            {"generator": state.generator, "discriminator": state.discriminator},
            directory=os.path.abspath(directory or checkpoint_manager.directory),
        )
        [g_state, d_state] = operator.itemgetter("generator", "discriminator")(restored)
        return State(step=jnp.int32(step), generator=g_state, discriminator=d_state)

    if _CHECKPOINT_FROM.value and (checkpoint_steps := orbax.checkpoint.utils.checkpoint_steps(_CHECKPOINT_FROM.value)):
        logging.info(f"Found checkpoint directory {_CHECKPOINT_FROM.value} with steps {checkpoint_steps}.")
        latest_checkpoint_step = max(checkpoint_steps)
        if not latest_step or latest_checkpoint_step > latest_step:
            return _restore_state(latest_checkpoint_step, _CHECKPOINT_FROM.value)
    if latest_step:
        return _restore_state(latest_step)

    logging.info("No checkpoint found.")
    return state

jax.config.parse_flags_with_absl()


def _maybe_remove_absl_logger() -> None:
    if (absl_handler := absl_logging.get_absl_handler()) in logging.root.handlers:
        logging.root.removeHandler(absl_handler)


from dsm.state import FittedValueTrainState
import numpy.typing as npt
from dsm import datasets, plotting, rewards
from dsm.configs import Config
from dsm.plotting import utils as plot_utils
def compute_return_distribution(
    # compute_distribution_metrics
    state: FittedValueTrainState,
    rng: jax.Array,
    *,
    config: Config,
) -> tuple[dict[str, npt.NDArray], dict[str, float]]:
    policy = datasets.make_policy(config.env)

    dsm_returns = {}
    for reward_fn_name, reward_fn in getattr(rewards, config.env).items():
        logging.info(f"Computing distribution metrics for {config.env} reward function {reward_fn_name}")
        for _stuff, source in zip(*plotting.source_states(config.env)):
            jax.debug.print("DEBUG DISTR METRICS Source {bar}", bar=source)
            jax.debug.print("DEBUG DISTR METRICS Stuff {bar}", bar=_stuff)
            dsr_return_distribution = plot_utils.return_distribution(
                state,
                rng,
                source,
                policy=policy,
                reward_fn=reward_fn,
                num_samples=config.plot_num_samples,
                config=config,
            )
            # jax.debug.print("DEBUG DISTR METRICS Rewards {bar}", bar=reward_fn_name)
            # jax.debug.print("DEBUG DISTR METRICS Source {bar}", bar=source)
            # jax.debug.print("DEBUG DISTR METRICS dsr_return_distribution {bar}", bar=dsr_return_distribution)
            
            if reward_fn_name not in dsm_returns:
                dsm_returns[reward_fn_name] = [dsr_return_distribution.tolist()]
            else:
                dsm_returns[reward_fn_name].extend(dsr_return_distribution.tolist())
                
    return dsm_returns
            


def main(_) -> None:
    _maybe_remove_absl_logger()
        
    buildable = fdl_flags.create_buildable_from_flags(configs)
    
    # print('debug BUILDABLE')

    logging.info(printing.as_str_flattened(buildable))
    config: configs.Config = fdl.build(buildable)
    
    workdir: epath.Path = _WORKDIR.value
    workdir.mkdir(parents=True, exist_ok=True)

    jax.debug.print("DEBUG directory {bar}", bar=os.path.abspath(workdir))
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        # os.path.abspath(workdir),
        workdir,
        checkpointers={
            "generator": orbax.checkpoint.PyTreeCheckpointer(),
            "discriminator": orbax.checkpoint.PyTreeCheckpointer(),
        },
        options=orbax.checkpoint.CheckpointManagerOptions( max_to_keep=3, enable_async_checkpointing=False, async_options=None,create=True, ),
    )

    # def checkpoint_callback(state: State) -> None:
    #     jax.debug.print('DEBUG STEP 3.8')
    #     step = state.step.item()
    #     checkpoint_manager.save(
    #         step,
    #         {"generator": state.generator, "discriminator": state.discriminator},
    #         metrics={
    #             "generator": state.generator.metrics.compute(),
    #             "discriminator": state.generator.metrics.compute(),
    #         },
    #     )
    #     # checkpoint_manager.wait_until_finished()

    env = envs.make(config.env)
    env = stade.GymEnvWrapper(env, with_infos=False, seed=None)
    rng = np.random.default_rng(config.seed)

    # data = datasets.make_dataset(config.env)

    rng_key = jax.random.PRNGKey(rng.integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max))
    rng_key, state_rng_key = jax.random.split(rng_key)
    
    # for checkpointing
    state = train.make_state(state_rng_key, typing.cast(specs.DiscreteArray, env.observation_spec()), config)
    state = _maybe_restore_state(checkpoint_manager, state)
    print('Saved model: ')
    
    def print_params_shapes(params: Any, prefix: str = ""):
        if isinstance(params, dict):
            for key, value in params.items():
                print_params_shapes(value, f"{prefix}.{key}" if prefix else key)
        elif isinstance(params, (jax.Array, jnp.ndarray)):
            print(f"{prefix}: {params.shape}")
        else:
            print(f"{prefix}: {type(params)}")
    print("Generator params shapes:")
    print_params_shapes(state.generator.params)
    print("\nDiscriminator params shapes:")
    print_params_shapes(state.discriminator.params)
    
    def print_main_keys(obj):
        keys = vars(obj).keys()
        print("\nMain keys of state:")
        for key in keys:
            print(key)
    print_main_keys(state)
    
    dsm_returns = compute_return_distribution(
            state.generator, jax.random.PRNGKey(0), config=config
        )
    
    import pickle

    # Saving
    with open('dsm_returns.pkl', 'wb') as f:
        pickle.dump(dsm_returns, f)
        print('Saved dsm_returns')

    print('creatng plot')
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Loading
    with open('dsm_returns.pkl', 'rb') as f:
        dsm_returns = pickle.load(f)
        # for task, reward_distributions in dsm_returns.items():
        #     plt.figure()
        #     plt.title(f'Return Distribution for {task}')
        #     plt.hist(reward_distributions, bins=20, alpha=0.5)
        #     plt.legend()
        #     plt.xlabel('Return')
        #     plt.ylabel('Frequency')
        #     plt.show()
                
        num_plots = len(dsm_returns)
        num_cols = min(3, num_plots) 
        num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate rows needed
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 5*num_rows), sharey=True)
        fig.suptitle('Return Distribution Predictions - DSM', fontsize=14)
        # print(dsm_returns.items())
        for i, (title, returns) in enumerate(dsm_returns.items()):
            ax = axes.flatten()[i] if num_plots > 1 else axes # Handle single subplot case
            sns.histplot(returns, kde=True, color='blue', ax=ax)
            ax.set_title(title)
            ax.set_xlabel('Return')
            ax.set_ylabel('Density')
            ax.get_legend().remove()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for title 
        plt.savefig('dsm_returns.png')
        

if __name__ == "__main__":
    fiddle.extensions.jax.enable()

    logging.getLogger("jax").setLevel(logging.INFO)
    jax.config.update("jax_numpy_rank_promotion", "raise")

    app.run(main, flags_parser=fdl_flags.flags_parser)
